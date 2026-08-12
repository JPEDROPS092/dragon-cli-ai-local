"""Interactive, fancy (rich-based) main menu for Dragon CLI.

This module is a *presentation layer*: a modern, keyboard-navigable menu that
surfaces the local-LLM toolkit — chat, model listing, hardware-based
recommendations, the model hub, conversation history, and the HTTP server.

It deliberately stays a thin shell over the existing, fully-working modules
(``hardware``, ``recommendation``, ``runtime``, ``model_hub``, ``models``,
``chat``, ``storage``, ``app``). Everything the menu does can still be driven
from plain argparse flags (``dragon-cli --serve``, etc.), so the menu never
becomes a second, divergent code path.

Navigation:
    * Arrow keys (Up/Down) ........ move the highlight
    * Enter ........................ select the highlighted option
    * 1..7 ........................ jump straight to an option
    * / ........................... open the quick-command line
    * r ........................... refresh the hardware profile
    * q / Esc ..................... quit
"""

from __future__ import annotations

import asyncio
import copy
import os
import re
import sys
from urllib.parse import unquote, urlparse
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from . import __version__, app
from .config import DB_FILE, MODELS_DIR
from .hardware import detect_hardware
from .models import MODEL_CATALOG, add_model_to_catalog
from .recommendation import ModelRecommender
from .runtime import RuntimeConfig

# Palette — kept in one place so the look stays consistent. These are rich
# style/color fragments used inside markup, e.g. f"[{ACCENT}]text[/]".
ACCENT = "bright_cyan"
DIM = "grey58"
GOOD = "bright_green"
WARN = "bright_yellow"
BAD = "bright_red"


# --------------------------------------------------------------------------- #
# Keyboard input (POSIX + Windows)                                            #
# --------------------------------------------------------------------------- #
def _read_key() -> str:
    """Read a single keypress, decoding arrow-key escape sequences.

    Returns canonical tokens: ``"up"``, ``"down"``, ``"left"``, ``"right"``,
    ``"enter"``, ``"esc"``, or the literal character that was pressed.
    """
    # Windows: use the console API.
    if os.name == "nt":
        try:
            import msvcrt

            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):  # special-key prefix
                code = msvcrt.getwch()
                return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(code, ch)
            if ch in ("\r", "\n"):
                return "enter"
            return ch
        except Exception:
            return "enter"

    # POSIX: switch the tty to cbreak for a raw read, then restore.
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    if not sys.stdin.isatty():
        # Non-interactive stdin (piped/redirected): read a whole line and
        # collapse it to a single canonical key so the menu stays usable.
        line = sys.stdin.readline()
        if not line:           # EOF (e.g. Ctrl-D, or a closed pipe)
            return "esc"
        line = line.strip().lower()
        if not line:
            return "enter"
        return line[0]

    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        return "esc"           # stdin stopped being a tty mid-session: bail out
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
        if not ch:             # EOF on the tty
            return "esc"
        if ch == "\x1b":
            # Either a bare Esc or an escape sequence like ``\x1b[A`` (up).
            # Probe for more bytes without blocking forever.
            more: List[str] = []
            for _ in range(2):
                r, _, _ = select.select([fd], [], [], 0.05)
                if not r:
                    break
                more.append(sys.stdin.read(1))
            if more and more[0] == "[":
                return {"A": "up", "B": "down", "C": "right", "D": "left"}.get(
                    more[1] if len(more) > 1 else "", "esc"
                )
            return "esc"
        if ch in ("\r", "\n"):
            return "enter"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# --------------------------------------------------------------------------- #
# Menu model                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class MenuOption:
    key: str                            # numeric shortcut, e.g. "1"
    label: str                          # menu title
    description: str                    # one-line help text
    handler: Callable[[], bool]         # False => quit the menu, True => stay


class InteractiveMenu:
    """Owns the menu loop, cached hardware profile, and all action handlers."""

    # (key, label, description) — handler is bound by attribute name below.
    MENU_ITEMS: List[Tuple[str, str, str]] = [
        ("1", "Chat with model", "Talk to a local model with streaming"),
        ("2", "List models", "Browse local and remote models in the catalog"),
        ("3", "Auto recommendation", "Pick the best model for your hardware"),
        ("4", "Connect to hub", "Search and download models from Hugging Face"),
        ("5", "Manage history", "List, resume, export saved conversations"),
        ("6", "Serve HTTP", "Expose the model via an OpenAI-compatible API"),
        ("7", "Manage MCP servers", "Add, remove, and test MCP connectors"),
        ("8", "Quit", "Exit Dragon CLI"),
    ]

    # Slash shortcuts -> option key (or a special target).
    QUICK_COMMANDS: Dict[str, str] = {
        "/chat": "1", "/models": "2", "/rec": "3", "/recommend": "3",
        "/hub": "4", "/history": "5", "/serve": "6", "/mcp": "7",
        "/quit": "8", "/exit": "8", "/profile": "p", "/refresh": "r",
    }

    def __init__(self, args: Any, console: Optional[Console] = None) -> None:
        self.args = args
        self.console = console or Console()
        self.hardware: Dict[str, Any] = detect_hardware()
        self.recommender = ModelRecommender(self.hardware)
        self.runtime = RuntimeConfig(hardware_profile=self.hardware)
        self.selected = 0

    # -- small helpers ------------------------------------------------------ #
    def _pause(self, msg: str = "[Enter to return to menu]") -> None:
        """Block until Enter, used between an action and returning to the menu."""
        self.console.input(f"[{DIM}]{msg}[/]")

    def _splash(self) -> None:
        """Show the mascot icon + title once, before the live menu starts."""
        from .banner import ICON

        lines = ICON.splitlines()
        art_width = max((len(ln) for ln in lines), default=0)
        # Block-center: pad every line by the same amount so the picture keeps
        # its shape (rich's Align.center would mangle multi-line ASCII art).
        pad = max(0, (self.console.width - art_width) // 2)
        art = "\n".join(" " * pad + ln for ln in lines)
        self.console.print(Text(art, style=ACCENT))
        self.console.print(Align.center(Text.assemble(
            (" DRAGON CLI ", f"bold white on {ACCENT}"),
            (f"  v{__version__} ", DIM),
            ("  100% local LLM + MCP ", f"bold {GOOD}"),
        )))
        self.console.print()

    # -- public entry point ------------------------------------------------- #
    def run(self) -> None:
        """Main loop: show the menu, dispatch actions until the user quits."""
        self._splash()
        self.console.show_cursor(False)
        try:
            while True:
                choice = self._main_menu()
                if choice is None:          # user pressed q / Esc
                    break
                if choice.handler() is False:
                    break
        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            self.console.show_cursor(True)
            self.console.print(f"\n[{ACCENT}]Dragon asleep.[/] See you next time!\n")

    # ------------------------------------------------------------------ #
    # Rendering                                                          #
    # ------------------------------------------------------------------ #
    def _header(self) -> Panel:
        title = Text()
        title.append(" DRAGON CLI ", style=f"bold white on {ACCENT}")
        title.append(f"  v{__version__} ", style=DIM)
        title.append("  100% local ", style=f"bold {GOOD}")
        return Panel(Align.center(title), border_style=ACCENT, padding=(0, 1))

    def _status_grid(self) -> Table:
        hw = self.hardware
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style=f"bold {DIM}", no_wrap=True)
        grid.add_column()

        backend = self.runtime.decide_execution_mode().upper()
        backend_style = GOOD if backend == "GPU" else (WARN if backend == "HYBRID" else ACCENT)

        grid.add_row("System", str(hw.get("os", "?")))
        grid.add_row("CPU", f"{hw.get('cpu_model', '?')}  ({hw.get('cpu_count', '?')} cores)")

        ram_total = hw.get("ram_total_gb", 0) or 0
        ram_avail = hw.get("ram_available_gb", 0) or 0
        grid.add_row("RAM", f"{ram_avail:.1f} GB free / {ram_total:.1f} GB total")

        if hw.get("gpu_available"):
            grid.add_row(
                "GPU",
                Text(f"{hw.get('gpu_name', '?')}  ({hw.get('gpu_vram_gb', 0):.1f} GB VRAM)", style=GOOD),
            )
        else:
            grid.add_row("GPU", Text("not detected (CPU mode)", style=DIM))

        grid.add_row("Backend", Text(backend, style=backend_style))

        best = self.recommender.recommend_best()
        if best:
            score = best.get("compatibility_score", 0)
            grid.add_row("Recommended", f"{best['name']}  [{DIM}]({score:.0%})[/]")
        else:
            grid.add_row("Recommended", f"[{DIM}]none compatible[/]")
        return grid

    def _status_panel(self) -> Panel:
        return Panel(
            self._status_grid(),
            title=f"[bold {ACCENT}]System status[/]",
            border_style=DIM,
            padding=(1, 2),
        )

    def _menu_panel(self) -> Panel:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold", width=3)
        table.add_column(no_wrap=True)
        table.add_column()

        for i, (key, label, desc) in enumerate(self.MENU_ITEMS):
            if i == self.selected:
                marker = f"[bold {ACCENT}]›[/]"
                label_txt = f"[bold white on {ACCENT}] {label} [/]"
                desc_txt = f"[{DIM}]{desc}[/]"
            else:
                marker = " "
                label_txt = f"[bold]{label}[/]"
                desc_txt = f"[{DIM}]{desc}[/]"
            table.add_row(f"[{DIM}]{key}[/]", f"{marker} {label_txt}", desc_txt)
        return Panel(
            Align.center(table),
            title=f"[bold {ACCENT}]Main menu[/]",
            border_style=ACCENT,
            padding=(1, 2),
        )

    def _footer(self) -> Text:
        return Align.center(Text.assemble(
            ("Arrows", f"bold {ACCENT}"), (" move  ", DIM),
            ("Enter", f"bold {ACCENT}"), (" select  ", DIM),
            ("1-8", f"bold {ACCENT}"), (" shortcut  ", DIM),
            ("/", f"bold {ACCENT}"), (" commands  ", DIM),
            ("r", f"bold {ACCENT}"), (" refresh  ", DIM),
            ("q", f"bold {ACCENT}"), (" quit", DIM),
        ))

    def _frame(self) -> Group:
        return Group(
            self._header(),
            Text(),  # spacer
            self._status_panel(),
            Text(),
            self._menu_panel(),
            Text(),
            self._footer(),
        )

    def _main_menu(self) -> Optional[MenuOption]:
        """Render the live menu and return the option the user picked."""
        options = [
            MenuOption(k, l, d, getattr(self, f"_action_{k}"))
            for k, l, d in self.MENU_ITEMS
        ]
        self.selected = max(0, min(self.selected, len(options) - 1))
        with Live(self._frame(), console=self.console, refresh_per_second=12, screen=False) as live:
            while True:
                key = _read_key()
                if key in ("up", "k"):
                    self.selected = (self.selected - 1) % len(options)
                elif key in ("down", "j"):
                    self.selected = (self.selected + 1) % len(options)
                elif key == "enter":
                    return options[self.selected]
                elif key in ("q", "Q", "esc"):
                    return None
                elif key == "r":
                    self._refresh()
                elif key == "/":
                    target = self._quick_command()
                    if target in (None, "p", "r"):
                        if target == "p":
                            self._show_profile()
                        elif target == "r":
                            self._refresh()
                    else:
                        idx = int(target) - 1
                        if 0 <= idx < len(options):
                            self.selected = idx
                            return options[idx]
                elif key.isdigit() and 1 <= int(key) <= len(options):
                    self.selected = int(key) - 1
                    return options[self.selected]
                live.update(self._frame())

    # ------------------------------------------------------------------ #
    # Action handlers                                                    #
    # ------------------------------------------------------------------ #
    def _refresh(self) -> None:
        """Re-detect hardware and rebuild the recommender/runtime."""
        self.hardware = detect_hardware()
        self.recommender = ModelRecommender(self.hardware)
        self.runtime = RuntimeConfig(hardware_profile=self.hardware)

    def _quick_command(self) -> Optional[str]:
        """Pop a quick-command line and resolve it to a target option key."""
        self.console.show_cursor(True)
        cmd = Prompt.ask(
            f"\n[{ACCENT}]Quick command[/] [{DIM}](/help to list)[/]",
            console=self.console, default="",
        ).strip().lower()
        self.console.show_cursor(False)
        if not cmd:
            return None
        if cmd in ("/help", "/?"):
            cmds = "  ".join(sorted(self.QUICK_COMMANDS))
            self.console.print(Panel(
                cmds, title=f"[{ACCENT}]Quick commands[/]", border_style=DIM))
            self._pause("[Enter to continue]")
            return None
        target = self.QUICK_COMMANDS.get(cmd)
        if target is None:
            self.console.print(f"[{BAD}]Unknown command:[/] {cmd}")
            self._pause("[Enter to continue]")
        return target

    def _require_llm(self) -> bool:
        """Check that llama-cpp-python is importable, else show how to install."""
        try:
            from .models.loader import _preload_cuda_runtime
            _preload_cuda_runtime()
            import llama_cpp  # noqa: F401
            return True
        except ImportError:
            self.console.print(Panel(
                "[bold red]llama-cpp-python is not installed.[/]\n\n"
                "Install the inference backend with:\n"
                f"  [{ACCENT}]dragon-cli-install-llm[/]\n\n"
                "(detects your OS/GPU and uses the most suitable prebuilt wheel)",
                title=f"[{BAD}]Missing dependency[/]", border_style=BAD))
            self._pause()
            return False

    def _run_app(self, **overrides: Any) -> None:
        """Run the async ``app.run`` orchestration with a copy of the args."""
        a = copy.copy(self.args)
        for k, v in overrides.items():
            setattr(a, k, v)
        try:
            asyncio.run(app.run(a))
        except (KeyboardInterrupt, EOFError):
            pass

    def _ask_gpu(self) -> Dict[str, Any]:
        """Ask how to run the model on the backend; return ``_run_app`` overrides.

        Mirrors the ``--gpu``/``--cpu``/``--n-gpu-layers`` CLI flags. When no
        GPU is detected the choice is skipped and CPU is forced.
        """
        if not self.hardware.get("gpu_available"):
            return {"cpu": True, "gpu": False, "n_gpu_layers": -1}
        choice = Prompt.ask(
            f"[{ACCENT}]Backend[/] [{DIM}](how to run the model)[/]",
            choices=["auto", "gpu", "cpu"], default="auto", console=self.console,
        )
        return {
            "cpu": choice == "cpu",
            "gpu": choice == "gpu",
            "n_gpu_layers": -1,
        }

    # 1 - Chat with model ---------------------------------------------- #
    def _action_1(self) -> bool:
        if not self._require_llm():
            return True
        pick = self._pick_model(title="Choose a model to chat with")
        if pick is None:
            return True
        idx, _model = pick
        self._run_app(serve=False, model=idx, **self._ask_gpu())
        self._pause()
        return True

    # 2 - List models -------------------------------------------------- #
    def _action_2(self) -> bool:
        self._show_models_table()
        self._pause()
        return True

    # 3 - Auto recommendation ------------------------------------------ #
    def _action_3(self) -> bool:
        self._show_recommendations()
        pick = IntPrompt.ask(
            f"\n[{ACCENT}]Use which model number?[/] [{DIM}](0 = back)[/]",
            console=self.console, default=0,
        )
        if pick == 0:
            return True
        if not (1 <= pick <= len(MODEL_CATALOG)):
            self.console.print(f"[{BAD}]Number out of catalog range.[/]")
            self._pause()
            return True
        if not self._require_llm():
            return True
        self._run_app(serve=False, model=pick)
        self._pause()
        return True

    # 4 - Connect to hub ----------------------------------------------- #
    def _action_4(self) -> bool:
        self._show_models_table(title="Connect to hub — catalog")
        choice_raw = Prompt.ask(
            f"\n[{ACCENT}]Hub action[/] [{DIM}](catalog # download, 's' search/add, 'hf://' add direct, Enter = back)[/]",
            console=self.console, default="",
        ).strip()
        choice = choice_raw.lower()
        if choice in ("", "0"):
            return True
        if choice in ("s", "search"):
            model = self._hub_search()
            if model:
                idx, created = self._add_model_to_catalog(model)
                action = "Added" if created else "Already in catalog"
                self.console.print(f"[{GOOD}]{action}[/] as catalog #{idx}: {model['name']}")
                if Confirm.ask("Download this model now?", console=self.console, default=True):
                    self._download_model_entry(model)
        elif choice.isdigit() and 1 <= int(choice) <= len(MODEL_CATALOG):
            self._download_model(int(choice))
        else:
            model = self._build_model_from_hf_reference(choice_raw)
            if model:
                idx, created = self._add_model_to_catalog(model)
                action = "Added" if created else "Already in catalog"
                self.console.print(f"[{GOOD}]{action}[/] as catalog #{idx}: {model['name']}")
                if Confirm.ask("Download this model now?", console=self.console, default=True):
                    self._download_model_entry(model)
            else:
                self.console.print(f"[{BAD}]Invalid option.[/]")
        self._pause()
        return True

    # 5 - Manage history ----------------------------------------------- #
    def _action_5(self) -> bool:
        self._manage_history()
        return True

    # 6 - Serve HTTP --------------------------------------------------- #
    def _action_6(self) -> bool:
        if not self._require_llm():
            return True
        host = Prompt.ask("Host", console=self.console, default=getattr(self.args, "host", "127.0.0.1"))
        port = IntPrompt.ask("Port", console=self.console, default=getattr(self.args, "port", 8000))
        gpu = self._ask_gpu()
        pick = self._pick_model(title=f"Model to serve at http://{host}:{port}/v1")
        if pick is None:
            return True
        idx, _model = pick
        self._run_app(serve=True, model=idx, host=host, port=port, **gpu)
        self._pause()
        return True

    # 7 - Manage MCP servers ------------------------------------------- #
    def _action_7(self) -> bool:
        self._manage_mcp()
        return True

    # 8 - Quit --------------------------------------------------------- #
    def _action_8(self) -> bool:
        return False

    # ------------------------------------------------------------------ #
    # MCP management                                                     #
    # ------------------------------------------------------------------ #
    def _manage_mcp(self) -> None:
        from .mcp import MCP_PRESETS, load_mcp_config, save_mcp_config

        config = load_mcp_config()
        servers = dict(config.get("mcpServers", {}))

        while True:
            self.console.print(Panel(
                f"Config file: [{ACCENT}]{MCP_CONFIG_FILE}[/]",
                title=f"[bold {ACCENT}]MCP servers[/]", border_style=ACCENT))

            if not servers:
                self.console.print(f"[{DIM}]No MCP servers configured.[/]")
            else:
                table = Table(border_style=DIM, expand=True)
                table.add_column("Name", style=f"bold {ACCENT}")
                table.add_column("Command", overflow="fold")
                table.add_column("Args", overflow="fold")
                for name, cfg in servers.items():
                    table.add_row(name, cfg.get("command", "?"),
                                  " ".join(str(a) for a in cfg.get("args", [])))
                self.console.print(table)

            action = Prompt.ask(
                f"\n[{ACCENT}]Action[/] [{DIM}](add, remove, test, back)[/]",
                console=self.console, default="",
            ).strip().lower()
            if not action or action in ("back", "b", "q"):
                return
            if action in ("add", "a"):
                self._mcp_add(servers, MCP_PRESETS)
                save_mcp_config(servers)
            elif action in ("remove", "r", "delete", "d"):
                self._mcp_remove(servers)
                save_mcp_config(servers)
            elif action in ("test", "t"):
                self._mcp_test(servers)
            else:
                self.console.print(f"[{BAD}]Unknown action.[/]")
            self._pause("[Enter to continue]")
            self.console.print()

    def _mcp_add(self, servers: dict, presets: dict) -> None:
        preset_names = sorted(presets)
        self.console.print(f"[{DIM}]Presets:[/] {', '.join(preset_names)}  "
                           f"(or 'custom' for your own command)")
        choice = Prompt.ask(
            f"[{ACCENT}]Server to add[/]", console=self.console, default="",
        ).strip().lower()
        if not choice:
            return
        if choice in presets:
            name = choice
            if name in servers and not Confirm.ask(
                    f"'{name}' already exists. Overwrite with preset?",
                    console=self.console, default=False):
                return
            servers[name] = {
                "command": presets[choice]["command"],
                "args": list(presets[choice]["args"]),
            }
            self.console.print(f"[{GOOD}]Added preset '{name}':[/] {presets[choice]['description']}")
            return
        if choice in ("custom", "c"):
            name = Prompt.ask(f"[{ACCENT}]Name for this server[/]", console=self.console).strip()
            if not name:
                return
            if name in servers and not Confirm.ask(
                    f"'{name}' exists. Overwrite?", console=self.console, default=False):
                return
            command = Prompt.ask(f"[{ACCENT}]Command[/] [{DIM}](e.g. npx, uvx, python)[/]",
                                 console=self.console, default="uvx")
            args_raw = Prompt.ask(
                f"[{ACCENT}]Arguments[/] [{DIM}](space-separated)[/]",
                console=self.console, default="",
            )
            servers[name] = {"command": command, "args": args_raw.split()}
            self.console.print(f"[{GOOD}]Added '{name}'.[/]")
            return
        self.console.print(f"[{BAD}]Unknown choice '{choice}'.[/]")

    def _mcp_remove(self, servers: dict) -> None:
        if not servers:
            self.console.print(f"[{DIM}]Nothing to remove.[/]")
            return
        name = Prompt.ask(
            f"[{ACCENT}]Server name to remove[/] [{DIM}](one of: {', '.join(servers)})[/]",
            console=self.console, default="",
        ).strip()
        if name in servers:
            del servers[name]
            self.console.print(f"[{GOOD}]Removed '{name}'.[/]")
        else:
            self.console.print(f"[{BAD}]No server named '{name}'.[/]")

    def _mcp_test(self, servers: dict) -> None:
        import asyncio

        if not servers:
            self.console.print(f"[{DIM}]No servers to test.[/]")
            return
        name = Prompt.ask(
            f"[{ACCENT}]Server to test[/] [{DIM}](Enter = all)[/]",
            console=self.console, default="",
        ).strip()
        targets = {name: servers[name]} if name in servers else servers
        asyncio.run(self._mcp_test_targets(targets))

    async def _mcp_test_targets(self, targets: dict) -> None:
        from contextlib import AsyncExitStack

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            self.console.print(f"[{BAD}]Package 'mcp' is not installed. "
                               f"Run: pip install mcp[/]")
            return

        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            for name, cfg in targets.items():
                try:
                    params = StdioServerParameters(
                        command=cfg["command"], args=cfg.get("args", []), env=cfg.get("env"),
                    )
                    read, write = await stack.enter_async_context(stdio_client(params))
                    session = await stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                    result = await session.list_tools()
                    self.console.print(
                        f"[{GOOD}]\u2713 {name}[/] connected ({len(result.tools)} tools)")
                except Exception as exc:  # noqa: BLE001
                    self.console.print(f"[{BAD}]\u2717 {name}[/] failed: {exc}")
        finally:
            await stack.aclose()

    # ------------------------------------------------------------------ #
    # Rich helper views                                                  #
    # ------------------------------------------------------------------ #
    def _pick_model(self, title: str) -> Optional[Tuple[int, dict]]:
        """Render the catalog and ask for a number. Returns (1-based idx, model)."""
        self._show_models_table(title=title)
        choice = IntPrompt.ask(
            f"\n[{ACCENT}]Model number[/] [{DIM}](0 = back)[/]",
            console=self.console, default=0,
        )
        if choice == 0 or not (1 <= choice <= len(MODEL_CATALOG)):
            if choice != 0:
                self.console.print(f"[{BAD}]Number out of catalog range.[/]")
                self._pause("[Enter]")
            return None
        return choice, MODEL_CATALOG[choice - 1]

    def _show_models_table(self, title: str = "Models — local and remote") -> None:
        from .models.catalog import available_ram_gb

        ram = available_ram_gb()
        vram = self.hardware.get("gpu_vram_gb", 0) or 0
        has_gpu = self.hardware.get("gpu_available", False)
        table = Table(title=f"[bold {ACCENT}]{title}[/]", border_style=ACCENT, expand=True)
        table.add_column("#", style=DIM, width=3)
        table.add_column("Model", overflow="fold")
        table.add_column("Min RAM", justify="right")
        table.add_column("Context", justify="right")
        table.add_column("GPU", justify="center")
        table.add_column("Source")

        for i, m in enumerate(MODEL_CATALOG, 1):
            downloaded = (MODELS_DIR / m["filename"]).exists()
            source = f"[{GOOD}]local[/]" if downloaded else f"[{DIM}]remote[/]"
            ram_warn = f" [{BAD}]![/]" if 0 < ram < m["ram_min_gb"] else ""
            need = m.get("vram_min_gb")
            if not has_gpu or not need:
                gpu_cell = f"[{DIM}]-[/]"
            elif vram >= need:
                gpu_cell = f"[{GOOD}]\u2713 {need:.0f}G[/]"
            elif vram >= need * 0.5:
                gpu_cell = f"[{WARN}]~ {need:.0f}G[/]"
            else:
                gpu_cell = f"[{BAD}]\u2717 {need:.0f}G[/]"
            table.add_row(
                str(i), m["name"],
                f"{m['ram_min_gb']} GB{ram_warn}", f"{m['ctx']}", gpu_cell, source,
            )
        self.console.print(table)
        gpu_line = (f"  |  GPU VRAM: {vram:.1f} GB free"
                    if has_gpu else "  |  GPU: none")
        self.console.print(f"[{DIM}]Free RAM detected: {ram:.1f} GB{gpu_line}  "
                           f"(\u2713 fits GPU fully, ~ partial, \u2717 too large)[/]\n")

    def _show_recommendations(self) -> None:
        compatible = self.recommender.get_compatible_models()
        if not compatible:
            self.console.print(Panel(
                f"[{BAD}]No models compatible with your hardware.[/]",
                border_style=BAD))
            return

        table = Table(
            title=f"[bold {ACCENT}]Auto recommendation[/]",
            border_style=ACCENT, expand=True,
        )
        table.add_column("#", style=DIM, width=3)
        table.add_column("Model", overflow="fold")
        table.add_column("Fit")
        table.add_column("Min RAM", justify="right")
        table.add_column("Backend", justify="center")

        for i, m in enumerate(compatible, 1):
            score = m.get("compatibility_score", 0)
            bar_len = 10
            filled = int(round(score * bar_len))
            color = GOOD if score >= 0.7 else (WARN if score >= 0.4 else BAD)
            bar = f"[{color}]{'█' * filled}[/][{DIM}]{'░' * (bar_len - filled)}[/] {score:.0%}"
            backend = f"[{GOOD}]GPU[/]" if m.get("can_use_gpu") else f"[{ACCENT}]CPU[/]"
            tag = f" [{GOOD}][recommended][/]" if m.get("recommended") else ""
            table.add_row(str(i), m["name"] + tag, bar, f"{m.get('ram_min_gb', '?')} GB", backend)

        self.console.print(table)
        best = compatible[0]
        self.console.print(Panel(
            self.recommender.get_recommendation_reason(best),
            title=f"[{ACCENT}]Why {best['name']}?[/]", border_style=DIM))
        self.console.print(f"[{DIM}]Use the number from the table above (1 = best option).[/]\n")

    def _download_model(self, idx: int) -> None:
        self._download_model_entry(MODEL_CATALOG[idx - 1])

    def _download_model_entry(self, model: dict) -> None:
        from .models.loader import download_model

        dest = MODELS_DIR / model["filename"]
        if dest.exists():
            self.console.print(f"[{GOOD}]Model already saved:[/] {dest}")
            return
        self.console.print(f"\n[{ACCENT}]Downloading[/] {model['name']} ...")
        with self.console.status(f"[{ACCENT}]Downloading {model['filename']}[/]", spinner="dots"):
            try:
                path = download_model(model)
            except Exception as exc:
                self.console.print(f"[{BAD}]Download failed:[/] {exc}")
                return
        self.console.print(f"[{GOOD}]Saved at[/] {path}")

    def _hub_search(self) -> Optional[dict]:
        from .model_hub import ModelHub

        query = Prompt.ask(
            f"[{ACCENT}]Search Hugging Face[/] [{DIM}](query, owner/repo, hf://, or HF URL)[/]",
            console=self.console, default="GGUF",
        ).strip()
        direct_model = self._build_model_from_hf_reference(query)
        if direct_model is not None:
            return direct_model

        hub = ModelHub()
        with self.console.status(f"[{ACCENT}]Searching '{query}'...[/]", spinner="dots"):
            try:
                results = hub.fetch_huggingface_models(search_query=query, limit=15)
            except Exception as exc:
                self.console.print(f"[{BAD}]Search error:[/] {exc}")
                return None
        if not results:
            self.console.print(
                f"[{DIM}]No results.[/] Hugging Face search matches repositories, not always exact file names."
            )
            if query.lower().endswith(".gguf"):
                self.console.print(
                    f"[{DIM}]Tip:[/] use direct reference, e.g. "
                    f"hf://owner/repo/{query} or paste the full HF resolve URL."
                )
            return None
        table = Table(title=f"[bold {ACCENT}]Hugging Face: '{query}'[/]", border_style=ACCENT)
        table.add_column("#", style=DIM, width=3)
        table.add_column("Repository", overflow="fold")
        table.add_column("Downloads", justify="right")
        table.add_column("Likes", justify="right")
        for i, m in enumerate(results, 1):
            table.add_row(
                str(i),
                m.get("name", m.get("repo_id", "?")),
                f"{m.get('downloads', 0):,}",
                f"{m.get('likes', 0):,}",
            )
        self.console.print(table)
        repo_choice = IntPrompt.ask(
            f"\n[{ACCENT}]Repository number[/] [{DIM}](0 = back)[/]",
            console=self.console,
            default=0,
        )
        if repo_choice == 0 or not (1 <= repo_choice <= len(results)):
            return None

        selected = results[repo_choice - 1]
        repo_id = selected.get("repo_id") or selected.get("name")
        if not repo_id:
            self.console.print(f"[{BAD}]Selected result has no repo id.[/]")
            return None

        return self._build_model_from_repo(repo_id)

    def _build_model_from_repo(self, repo_id: str) -> Optional[dict]:
        from .model_hub import ModelHub

        hub = ModelHub()
        with self.console.status(f"[{ACCENT}]Loading GGUF files from {repo_id}...[/]", spinner="dots"):
            files = hub.get_model_files(repo_id)
        if not files:
            self.console.print(f"[{BAD}]No .gguf files found in[/] {repo_id}")
            return None

        file_table = Table(title=f"[bold {ACCENT}]GGUF files — {repo_id}[/]", border_style=ACCENT)
        file_table.add_column("#", style=DIM, width=3)
        file_table.add_column("Filename", overflow="fold")
        file_table.add_column("Quant", justify="center")
        for i, fn in enumerate(files, 1):
            file_table.add_row(str(i), fn, self._extract_quantization(fn))
        self.console.print(file_table)

        recommended_idx = self._pick_recommended_file_index(files)
        self.console.print(
            f"[{DIM}]Tip:[/] recommended file #{recommended_idx} for general local use."
        )

        file_choice = IntPrompt.ask(
            f"\n[{ACCENT}]File number[/] [{DIM}](0 = back)[/]",
            console=self.console,
            default=recommended_idx,
        )
        if file_choice == 0 or not (1 <= file_choice <= len(files)):
            return None

        filename = files[file_choice - 1]
        return self._build_model_entry(repo_id, filename)

    def _build_model_entry(self, repo_id: str, filename: str) -> dict:
        default_name = f"{repo_id.split('/')[-1]} ({self._extract_quantization(filename)}) — hub"
        model_name = Prompt.ask(
            f"[{ACCENT}]Catalog display name[/]",
            console=self.console,
            default=default_name,
        ).strip() or default_name

        params_b = self._estimate_params_billions(f"{repo_id} {filename}")
        ram_default = self._estimate_ram_default_gb(params_b, filename)
        vram_default = max(1.0, round(params_b * 0.7, 1))
        layers_default = self._estimate_layer_count(params_b)

        ram_min = IntPrompt.ask("Min RAM (GB)", console=self.console, default=ram_default)
        ctx = IntPrompt.ask("Context window", console=self.console, default=8192)

        return {
            "name": model_name,
            "repo_id": repo_id,
            "filename": filename,
            "ram_min_gb": max(1, ram_min),
            "vram_min_gb": vram_default,
            "n_layers": layers_default,
            "ctx": max(512, ctx),
        }

    def _add_model_to_catalog(self, model: dict) -> Tuple[int, bool]:
        return add_model_to_catalog(model, persist=True)

    @staticmethod
    def _estimate_params_billions(text: str) -> float:
        match = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", text)
        if not match:
            return 7.0
        try:
            return float(match.group(1))
        except ValueError:
            return 7.0

    @staticmethod
    def _estimate_layer_count(params_b: float) -> int:
        if params_b <= 2:
            return 24
        if params_b <= 4:
            return 28
        if params_b <= 8:
            return 32
        if params_b <= 14:
            return 40
        return 48

    @staticmethod
    def _extract_quantization(filename: str) -> str:
        upper = filename.upper()
        quant_match = re.search(r"Q\d(?:_[A-Z0-9]+)*", upper)
        return quant_match.group(0) if quant_match else "unknown"

    def _build_model_from_hf_uri(self, uri: str) -> Optional[dict]:
        """Build a catalog entry from ``hf://repo_id/filename.gguf``."""
        raw = uri.strip()
        if not raw.lower().startswith("hf://"):
            self.console.print(f"[{BAD}]Invalid Hugging Face URI.[/]")
            return None
        payload = raw[5:]
        if "/" not in payload:
            self.console.print(f"[{BAD}]Expected format:[/] hf://owner/repo/file.gguf")
            return None

        parts = [p for p in payload.split("/") if p]
        if len(parts) < 3:
            self.console.print(f"[{BAD}]Expected format:[/] hf://owner/repo/file.gguf")
            return None

        owner = parts[0]
        repo = parts[1]
        filename = "/".join(parts[2:])
        if not filename.lower().endswith(".gguf"):
            self.console.print(f"[{BAD}]Only .gguf files are supported here.[/]")
            return None

        repo_id = f"{owner}/{repo}"
        return self._build_model_entry(repo_id, filename)

    def _build_model_from_hf_reference(self, raw: str) -> Optional[dict]:
        """Accept hf:// URIs or Hugging Face resolve/blob URLs and build a model.

        Returns None when ``raw`` doesn't look like a direct model reference.
        """
        text = raw.strip()
        if not text:
            return None

        if text.lower().startswith("hf://"):
            return self._build_model_from_hf_uri(text)

        if text.lower().startswith("http://") or text.lower().startswith("https://"):
            parsed = urlparse(text)
            if "huggingface.co" not in (parsed.netloc or "").lower():
                return None
            parts = [unquote(p) for p in parsed.path.split("/") if p]
            if len(parts) >= 2:
                owner, repo = parts[0], parts[1]
                if len(parts) >= 5:
                    marker = parts[2].lower()
                    if marker in ("resolve", "blob"):
                        filename = "/".join(parts[4:])
                        if filename.lower().endswith(".gguf"):
                            return self._build_model_from_hf_uri(f"hf://{owner}/{repo}/{filename}")
                return self._build_model_from_repo(f"{owner}/{repo}")

        snippet_repo = self._extract_repo_id_from_text(text)
        if snippet_repo:
            return self._build_model_from_repo(snippet_repo)

        if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", text):
            return self._build_model_from_repo(text)

        return None

    @staticmethod
    def _extract_repo_id_from_text(text: str) -> Optional[str]:
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", text)
        for token in quoted:
            if re.match(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", token):
                return token
        inline = re.search(r"\b([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)\b", text)
        if inline:
            return inline.group(1)
        return None

    @staticmethod
    def _estimate_ram_default_gb(params_b: float, filename: str) -> int:
        quant = filename.upper()
        # Conservative but closer to practical defaults for GGUF quantizations.
        if "Q2" in quant or "IQ2" in quant:
            factor = 0.45
        elif "Q3" in quant or "IQ3" in quant:
            factor = 0.6
        elif "Q4" in quant or "IQ4" in quant:
            factor = 0.8
        elif "Q5" in quant:
            factor = 1.0
        elif "Q6" in quant:
            factor = 1.15
        elif "Q8" in quant:
            factor = 1.45
        else:
            factor = 0.9
        return max(2, int(round(max(1.5, params_b * factor))))

    def _pick_recommended_file_index(self, files: List[str]) -> int:
        ranked = sorted(
            enumerate(files, start=1),
            key=lambda item: self._file_rank(item[1]),
        )
        return ranked[0][0] if ranked else 1

    @staticmethod
    def _file_rank(filename: str) -> Tuple[int, int, str]:
        upper = filename.upper()
        # Prefer common local-friendly quantizations first.
        quant_priority = [
            "Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M", "Q5_0", "Q3_K_M", "Q8_0",
        ]
        for i, q in enumerate(quant_priority):
            if q in upper:
                return (i, len(filename), upper)
        return (len(quant_priority), len(filename), upper)

    def _show_profile(self) -> None:
        hw = self.hardware
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style=f"bold {DIM}")
        grid.add_column()
        for label, value in (
            ("System", hw.get("os", "?")),
            ("CPU", hw.get("cpu_model", "?")),
            ("Cores (logical)", hw.get("cpu_count", "?")),
            ("Total RAM", f"{hw.get('ram_total_gb', 0):.1f} GB"),
            ("Free RAM", f"{hw.get('ram_available_gb', 0):.1f} GB"),
            ("Free disk", f"{hw.get('disk_free_gb', 0):.1f} GB"),
            ("GPU", hw.get("gpu_name", "not detected")),
            ("VRAM", f"{hw.get('gpu_vram_gb', 0):.1f} GB"),
        ):
            grid.add_row(label, str(value))
        self.console.print(Panel(
            grid, title=f"[bold {ACCENT}]Hardware profile[/]", border_style=ACCENT))
        self._pause("[Enter to continue]")

    def _manage_history(self) -> None:
        from pathlib import Path

        from .storage import ConversationStore

        store = ConversationStore(Path(getattr(self.args, "db", DB_FILE)))
        try:
            convs = store.list()
            if not convs:
                self.console.print(Panel(
                    f"[{DIM}]No saved conversations yet.[/]", border_style=DIM))
                self._pause()
                return

            table = Table(title=f"[bold {ACCENT}]Saved conversations[/]", border_style=ACCENT, expand=True)
            table.add_column("ID", style=DIM, width=5)
            table.add_column("Title", overflow="fold")
            table.add_column("Model", overflow="fold")
            table.add_column("Msgs", justify="right")
            table.add_column("Updated")
            for c in convs:
                table.add_row(
                    f"#{c.id}", c.title, c.model_name or "-",
                    str(c.message_count), c.updated_at,
                )
            self.console.print(table)

            choice = Prompt.ask(
                f"\n[{ACCENT}]Action[/] [{DIM}](resume <id>, delete <id>, export <id>, Enter = back)[/]",
                console=self.console, default="",
            ).strip()
            if not choice:
                return
            parts = choice.split()
            cmd = parts[0].lower()
            if cmd in ("resume", "r") and len(parts) == 2 and parts[1].isdigit():
                self._resume_conversation(store, int(parts[1]))
            elif cmd in ("delete", "d") and len(parts) == 2 and parts[1].isdigit():
                self._delete_conversation(store, int(parts[1]))
            elif cmd in ("export", "e") and len(parts) >= 2 and parts[1].isdigit():
                fmt = parts[2] if len(parts) > 2 and parts[2] in ("md", "json") else "md"
                self._export_conversation(store, int(parts[1]), fmt)
            else:
                self.console.print(
                    f"[{BAD}]Usage:[/] resume <id> | delete <id> | export <id> [md|json]")
        finally:
            store.close()
            self._pause()

    def _resume_conversation(self, store: Any, conv_id: int) -> None:
        if store.get(conv_id) is None:
            self.console.print(f"[{BAD}]Conversation #{conv_id} not found.[/]")
            return
        if not self._require_llm():
            return
        pick = self._pick_model(title="Model to resume the conversation")
        if pick is None:
            return
        idx, _model = pick
        self._run_app(serve=False, model=idx, resume=conv_id)

    def _delete_conversation(self, store: Any, conv_id: int) -> None:
        if not Confirm.ask(f"Delete conversation #{conv_id}?", console=self.console, default=False):
            self.console.print(f"[{DIM}]Cancelled.[/]")
            return
        if store.delete(conv_id):
            self.console.print(f"[{GOOD}]Conversation #{conv_id} deleted.[/]")
        else:
            self.console.print(f"[{BAD}]Conversation #{conv_id} not found.[/]")

    def _export_conversation(self, store: Any, conv_id: int, fmt: str) -> None:
        from pathlib import Path

        path = Path(f"conversation_{conv_id}.{fmt}")
        result = store.export(conv_id, path, fmt)
        if result is None:
            self.console.print(f"[{BAD}]Conversation #{conv_id} not found.[/]")
        else:
            self.console.print(f"[{GOOD}]Exported to[/] {result.resolve()}")


def run(args: Any) -> None:
    """Entry point called from ``cli.py`` when no action flag is given."""
    InteractiveMenu(args).run()


if __name__ == "__main__":
    import argparse

    run(argparse.Namespace(
        db=str(DB_FILE), host="127.0.0.1", port=8000,
        no_mcp=False, serve=False, model=None, resume=None,
        gpu=False, cpu=False, n_gpu_layers=-1,
    ))
