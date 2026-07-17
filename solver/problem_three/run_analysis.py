"""问题三入口：生成模拟包裹数据并计算 100 组 eta 场景。"""

try:
    from .eta_sensitivity import main as run_eta_sensitivity
    from .generate_packages import main as generate_packages
except ImportError:
    from eta_sensitivity import main as run_eta_sensitivity
    from generate_packages import main as generate_packages


def main() -> None:
    """依次生成包裹数据和 eta 敏感性分析结果。"""
    print("Step 1/2: Generating simulated package data...")
    generate_packages()
    print("\nStep 2/2: Solving 100 eta scenarios...")
    run_eta_sensitivity()
    print("\nProblem three analysis completed.")


if __name__ == "__main__":
    main()
