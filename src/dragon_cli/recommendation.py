"""Model recommendation engine for Dragon CLI.

Analyzes hardware profile and recommends optimal models based on:
- Available RAM and VRAM
- CPU capabilities
- User preferences (speed vs quality)
"""

from typing import Any, Dict, List, Optional, Tuple

from .models.catalog import MODEL_CATALOG
from .hardware import detect_hardware


class ModelRecommender:
    """Recommends LLM models based on hardware capabilities."""
    
    def __init__(self, hardware_profile: Optional[Dict[str, Any]] = None):
        """Initialize recommender with optional hardware profile.
        
        Args:
            hardware_profile: Hardware info dict from detect_hardware(),
                             or None to auto-detect
        """
        if hardware_profile is None:
            self.hardware = detect_hardware()
        else:
            self.hardware = hardware_profile
    
    def get_compatible_models(
        self,
        priority: str = "balanced",
        max_context: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get list of models compatible with current hardware.
        
        Args:
            priority: One of "speed", "quality", "balanced", "memory"
            max_context: Maximum context window size desired
            
        Returns:
            List of compatible models with scores
        """
        ram_available = self.hardware.get("ram_available_gb", 8)
        vram_available = self.hardware.get("gpu_vram_gb", 0)
        gpu_available = self.hardware.get("gpu_available", False)
        cpu_count = self.hardware.get("cpu_count", 4)
        
        compatible = []
        
        for model in MODEL_CATALOG:
            ram_required = model.get("ram_min_gb", 0)
            
            # Basic compatibility check
            if ram_required > ram_available * 1.2:  # Allow 20% overhead
                continue
            
            # Calculate compatibility score
            score = self._calculate_score(
                model=model,
                ram_available=ram_available,
                vram_available=vram_available,
                gpu_available=gpu_available,
                cpu_count=cpu_count,
                priority=priority,
            )
            
            model_copy = model.copy()
            model_copy["compatibility_score"] = score
            model_copy["can_use_gpu"] = gpu_available and vram_available >= 4
            model_copy["recommended"] = score >= 0.7
            
            compatible.append(model_copy)
        
        # Sort by score descending
        compatible.sort(key=lambda x: x["compatibility_score"], reverse=True)
        
        return compatible
    
    def _calculate_score(
        self,
        model: Dict[str, Any],
        ram_available: float,
        vram_available: float,
        gpu_available: bool,
        cpu_count: int,
        priority: str,
    ) -> float:
        """Calculate compatibility score for a model (0.0 to 1.0)."""
        ram_required = model.get("ram_min_gb", 0)
        
        # Base score: how well RAM fits
        ram_ratio = ram_available / max(ram_required, 1)
        ram_score = min(1.0, ram_ratio / 1.5)  # Cap at 1.0
        
        # GPU bonus
        gpu_score = 0.0
        if gpu_available:
            # Estimate VRAM needed (rough: 0.7 GB per billion params at Q4)
            params_estimate = self._estimate_params_from_name(model.get("name", ""))
            vram_needed = params_estimate * 0.7
            if vram_available >= vram_needed:
                gpu_score = 0.3  # Bonus for full GPU offload capability
            elif vram_available >= vram_needed * 0.5:
                gpu_score = 0.15  # Partial offload
        
        # CPU score (more cores = better for CPU inference)
        cpu_score = min(1.0, cpu_count / 8) * 0.1
        
        # Priority adjustments
        if priority == "speed":
            # Favor smaller models
            size_penalty = (ram_required / 16) * 0.2
            total_score = ram_score + gpu_score + cpu_score - size_penalty
        elif priority == "quality":
            # Favor larger models
            size_bonus = min(0.2, ram_required / 20)
            total_score = ram_score + gpu_score + cpu_score + size_bonus
        elif priority == "memory":
            # Strongly favor smaller models
            memory_efficiency = 1.0 - (ram_required / 20)
            total_score = ram_score * 0.5 + memory_efficiency * 0.5 + gpu_score
        else:  # balanced
            total_score = ram_score + gpu_score + cpu_score
        
        return min(1.0, max(0.0, total_score))
    
    def _estimate_params_from_name(self, name: str) -> float:
        """Estimate parameter count from model name (in billions)."""
        import re
        
        # Look for patterns like "7B", "14B", "1.5B", etc.
        matches = re.findall(r'(\d+\.?\d*)[Bb]', name)
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                pass
        return 7.0  # Default assumption
    
    def recommend_best(
        self,
        priority: str = "balanced",
    ) -> Optional[Dict[str, Any]]:
        """Get the single best recommended model.
        
        Args:
            priority: One of "speed", "quality", "balanced", "memory"
            
        Returns:
            Best model dictionary or None if no compatible models
        """
        compatible = self.get_compatible_models(priority=priority)
        if compatible:
            return compatible[0]
        return None
    
    def get_recommendation_reason(self, model: Dict[str, Any]) -> str:
        """Get human-readable explanation for why a model was recommended."""
        ram_available = self.hardware.get("ram_available_gb", 0)
        ram_required = model.get("ram_min_gb", 0)
        gpu_available = self.hardware.get("gpu_available", False)
        vram = self.hardware.get("gpu_vram_gb", 0)
        
        reasons = []
        
        # RAM assessment
        if ram_available >= ram_required * 1.5:
            reasons.append(f"✓ Plenty of RAM ({ram_available:.1f} GB available, {ram_required} GB needed)")
        elif ram_available >= ram_required:
            reasons.append(f"✓ Sufficient RAM ({ram_available:.1f} GB available)")
        else:
            reasons.append(f"⚠ Limited RAM ({ram_available:.1f} GB available, {ram_required} GB recommended)")
        
        # GPU assessment
        if gpu_available and vram >= 6:
            reasons.append(f"✓ GPU acceleration available ({vram:.1f} GB VRAM)")
        elif gpu_available:
            reasons.append(f"○ GPU available but limited VRAM ({vram:.1f} GB)")
        else:
            reasons.append("○ CPU-only mode (no GPU detected)")
        
        # Model-specific notes
        params = self._estimate_params_from_name(model.get("name", ""))
        if params <= 3:
            reasons.append("→ Fast inference, good for quick tasks")
        elif params <= 8:
            reasons.append("→ Good balance of speed and quality")
        else:
            reasons.append("→ Higher quality, slower inference")
        
        return "; ".join(reasons)
    
    def print_recommendations(
        self,
        top_n: int = 5,
        priority: str = "balanced",
    ) -> None:
        """Print formatted recommendations to console."""
        from .config import C_CYAN, C_RESET, C_GREEN, C_YELLOW, C_DIM, C_RED
        
        compatible = self.get_compatible_models(priority=priority)
        
        print(f"\n{C_CYAN}{'='*60}{C_RESET}")
        print(f"{C_CYAN}MODEL RECOMMENDATIONS{C_RESET}")
        print(f"{C_CYAN}{'='*60}{C_RESET}\n")
        
        # Show hardware summary
        print(f"{C_DIM}Your Hardware:{C_RESET}")
        print(f"  RAM: {self.hardware.get('ram_available_gb', 0):.1f} GB available")
        if self.hardware.get('gpu_available'):
            print(f"  GPU: {self.hardware.get('gpu_name', 'Unknown')} ({self.hardware.get('gpu_vram_gb', 0):.1f} GB VRAM)")
        else:
            print(f"  GPU: None detected (CPU-only)")
        print()
        
        if not compatible:
            print(f"{C_RED}No compatible models found for your hardware.{C_RESET}\n")
            return
        
        print(f"{C_DIM}Priority: {priority.capitalize()}{C_RESET}\n")
        print(f"{C_GREEN}Top Recommendations:{C_RESET}\n")
        
        for i, model in enumerate(compatible[:top_n], 1):
            score = model.get("compatibility_score", 0)
            recommended = model.get("recommended", False)
            
            badge = f"{C_GREEN}[RECOMMENDED]{C_RESET}" if recommended else f"{C_DIM}[Compatible]{C_RESET}"
            
            # Score indicator
            if score >= 0.8:
                score_str = f"{C_GREEN}██████████{C_RESET} {score:.0%}"
            elif score >= 0.6:
                score_str = f"{C_YELLOW}██████░░░░{C_RESET} {score:.0%}"
            else:
                score_str = f"{C_RED}████░░░░░░{C_RESET} {score:.0%}"
            
            print(f"  {i}. {model['name']}")
            print(f"     {badge}")
            print(f"     Match: {score_str}")
            print(f"     Min RAM: {model.get('ram_min_gb', '?')} GB | Context: {model.get('ctx', '?')}")
            
            if i == 1:
                reason = self.get_recommendation_reason(model)
                print(f"     {C_DIM}Why: {reason}{C_RESET}")
            print()


def recommend_model_for_hardware(
    priority: str = "balanced",
    show_all: bool = False,
) -> Optional[Dict[str, Any]]:
    """Convenience function to get model recommendation.
    
    Args:
        priority: One of "speed", "quality", "balanced", "memory"
        show_all: If True, print all recommendations
        
    Returns:
        Best recommended model or None
    """
    recommender = ModelRecommender()
    
    if show_all:
        recommender.print_recommendations(priority=priority)
    
    return recommender.recommend_best(priority=priority)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Get model recommendations for your hardware")
    parser.add_argument(
        "--priority",
        choices=["speed", "quality", "balanced", "memory"],
        default="balanced",
        help="Optimization priority (default: balanced)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all compatible models"
    )
    
    args = parser.parse_args()
    
    result = recommend_model_for_hardware(
        priority=args.priority,
        show_all=args.all
    )
    
    if result and not args.all:
        print(f"\n{result['name']}")
        print(f"Min RAM: {result.get('ram_min_gb', '?')} GB")
