"""
评估器模块

提供 Evaluator 类，封装测试集评估的完整流程：
1. 遍历测试集，用 model.generate() 贪心解码生成预测
2. 计算 Exact Match + Token Accuracy 指标
3. 生成可视化图表（训练曲线、预测样本、错误分析）
4. 输出格式化评估报告

与 Trainer.validate_epoch() 的区别：
- Trainer 在训练过程中评估验证集，重点在监控过拟合和早停
- Evaluator 在训练完成后评估测试集，重点在全面分析和报告
- Evaluator 不计算 Teacher Forcing loss（推理场景无意义），
  只关注自回归生成的准确率
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.paths import FIGURES_DIR
from src.data.mapping import VocabMapping
from src.evaluate.metrics import compute_metrics
from src.evaluate.visualize import (
    plot_error_by_length,
    plot_error_by_operator,
    plot_metrics_report,
    plot_prediction_samples,
    plot_training_history,
)


class Evaluator:
    """
    测试集评估器

    使用模型的自回归生成能力（model.generate()）在测试集上评估，
    收集预测结果、计算指标、生成可视化报告。

    使用方式:
        evaluator = Evaluator(model, test_loader, vocab, device)
        report = evaluator.evaluate(history={"train_loss": [...], ...})
    """

    def __init__(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        vocab: VocabMapping,
        device: torch.device,
    ):
        """
        Args:
            model:       已训练的 Seq2Seq 模型（eval 模式）
            test_loader: 测试集 DataLoader
            vocab:       词表映射，用于 decode 生成的 token 序列
            device:      计算设备
        """
        self.model = model
        self.test_loader = test_loader
        self.vocab = vocab
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        history: Optional[Dict[str, List[float]]] = None,
        output_dir: Path = FIGURES_DIR,
    ) -> dict:
        """
        执行完整评估流程

        步骤：
        1. 遍历测试集收集所有预测结果
        2. 计算 Exact Match + Token Accuracy
        3. 生成可视化图表
        4. 打印格式化报告

        Args:
            history:    训练历史字典（可选），传入则绘制训练曲线
            output_dir: 图表输出目录，默认 FIGURES_DIR

        Returns:
            评估报告字典：
            {
                "metrics": {...},           # compute_metrics() 的返回值
                "total_samples": int,       # 测试集总样本数
                "error_samples": [...],     # 错误样本列表
            }
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.eval()

        # ================================================================
        # Step 1: 收集所有预测结果
        # 遍历测试集，对每个 batch 用 model.generate() 生成预测，
        # 收集表达式、真实答案、预测答案三个列表
        # ================================================================
        all_expressions: List[str] = []
        all_predictions: List[str] = []
        all_targets: List[str] = []

        all_generated_ids: List[torch.Tensor] = []
        all_target_tensors: List[torch.Tensor] = []

        desc = "[Evaluate] Testing"
        for encoder_input, decoder_input, target_output, encoder_mask in tqdm(
            self.test_loader, desc=desc, unit="batch"
        ):
            encoder_input = encoder_input.to(self.device)
            encoder_mask = encoder_mask.to(self.device)

            batch_size = encoder_input.size(0)

            # 贪心解码生成
            generated_ids, _ = self.model.generate(
                encoder_input,
                encoder_mask,
                max_generation_length=target_output.size(1),
            )

            # 收集原始数据（用于可视化）
            for i in range(batch_size):
                # 编码器输入 -> 表达式字符串
                expression_ids = encoder_input[i].tolist()
                expression_string = self.vocab.decode(
                    expression_ids, strip_special=True
                )

                # 目标序列 -> 答案字符串
                target_ids = target_output[i].tolist()
                target_string = self.vocab.decode(target_ids, strip_special=True)

                # 生成序列 -> 预测答案字符串
                prediction_ids = generated_ids[i].tolist()
                prediction_string = self.vocab.decode(
                    prediction_ids, strip_special=True
                )

                all_expressions.append(expression_string)
                all_targets.append(target_string)
                all_predictions.append(prediction_string)

            # 收集张量数据（用于计算指标）
            all_generated_ids.append(generated_ids.cpu())
            all_target_tensors.append(target_output.cpu())

        # ================================================================
        # Step 2: 计算评估指标
        # ================================================================
        generated_concatenated = torch.cat(all_generated_ids, dim=0)
        target_concatenated = torch.cat(all_target_tensors, dim=0)

        # 如果生成序列比目标序列短，在右侧补 PAD 使维度一致
        max_generation_length = generated_concatenated.size(1)
        max_target_length = target_concatenated.size(1)
        if max_generation_length < max_target_length:
            padding = torch.full(
                (
                    generated_concatenated.size(0),
                    max_target_length - max_generation_length,
                ),
                self.vocab.pad_index,
                dtype=torch.long,
            )
            generated_concatenated = torch.cat([generated_concatenated, padding], dim=1)

        metrics = compute_metrics(
            generated_concatenated,
            target_concatenated,
            pad_index=self.vocab.pad_index,
            eos_index=self.vocab.eos_index,
        )

        # ================================================================
        # Step 3: 收集错误样本
        # ================================================================
        error_samples = []
        for i in range(len(all_expressions)):
            if all_predictions[i] != all_targets[i]:
                error_samples.append(
                    {
                        "expression": all_expressions[i],
                        "target": all_targets[i],
                        "prediction": all_predictions[i],
                    }
                )

        # ================================================================
        # Step 4: 生成可视化图表
        # ================================================================
        generated_plots: List[Path] = []

        # 训练历史曲线（仅当传入 history 时绘制）
        if history is not None and len(history.get("train_loss", [])) > 0:
            plot_path = plot_training_history(
                history,
                save_path=output_dir / "training_history.png",
            )
            generated_plots.append(plot_path)

        # 评估指标汇总
        plot_path = plot_metrics_report(
            metrics,
            save_path=output_dir / "metrics_report.png",
        )
        generated_plots.append(plot_path)

        # 预测样本对比
        plot_path = plot_prediction_samples(
            all_expressions,
            all_predictions,
            all_targets,
            save_path=output_dir / "prediction_samples.png",
        )
        generated_plots.append(plot_path)

        # 按表达式长度分组的准确率
        plot_path = plot_error_by_length(
            all_expressions,
            all_predictions,
            all_targets,
            save_path=output_dir / "error_by_length.png",
        )
        generated_plots.append(plot_path)

        # 按运算符分组的准确率
        plot_path = plot_error_by_operator(
            all_expressions,
            all_predictions,
            all_targets,
            save_path=output_dir / "error_by_operator.png",
        )
        generated_plots.append(plot_path)

        # ================================================================
        # Step 5: 打印格式化报告
        # ================================================================
        self._print_report(metrics, error_samples, generated_plots)

        return {
            "metrics": metrics,
            "total_samples": len(all_expressions),
            "error_samples": error_samples,
            "generated_plots": [str(p) for p in generated_plots],
        }

    def _print_report(
        self,
        metrics: dict,
        error_samples: List[dict],
        generated_plots: List[Path],
    ) -> None:
        """打印格式化的评估报告到控制台"""
        print()
        print("=" * 60)
        print("  评估报告")
        print("=" * 60)

        print(f"  测试样本数:      {metrics['exact_match_total']}")
        print(f"  完全正确数:      {metrics['exact_match_correct']}")
        print(f"  完全匹配准确率:  {metrics['exact_match'] * 100:.2f}%")
        print(f"  字符级准确率:    {metrics['token_accuracy'] * 100:.2f}%")
        print(f"  (有效 token 数:  {metrics['token_total']})")
        print()

        # 展示前 10 条错误样本
        error_count = len(error_samples)
        if error_count > 0:
            print(f"  错误样本数: {error_count}")
            print("  " + "-" * 56)
            print(f"  {'表达式':<15} {'真实答案':<12} {'模型预测':<12}")
            print("  " + "-" * 56)
            display_count = min(10, error_count)
            for sample in error_samples[:display_count]:
                print(
                    f"  {sample['expression']:<15} "
                    f"{sample['target']:<12} "
                    f"{sample['prediction']:<12}"
                )
            if error_count > display_count:
                print(f"  ... 还有 {error_count - display_count} 条错误")
            print()

        print(f"  图表输出: {len(generated_plots)} 个")
        for plot_path in generated_plots:
            print(f"    - {plot_path}")
        print("=" * 60)
        print()
