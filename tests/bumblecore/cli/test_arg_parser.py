import pytest
import sys
import os
import tempfile
import yaml
from unittest.mock import patch
from bumblecore.cli.arg_parser import get_args


class TestArgParser:

    def test_default_values(self):
        with patch.object(sys, 'argv', ['test_script.py']):
            args = get_args()
            
            assert args.yaml_config == ""
            assert args.cutoff_len == 1024
            assert args.num_epochs == 3.0
            assert args.learning_rate == 5e-5
            assert args.weight_decay == 0.01
            assert args.lr_scheduler_type == "cosine"
            assert args.warmup_ratio == 0.1
            assert args.train_micro_batch_size_per_gpu == 4
            assert args.gradient_accumulation_steps == 8
            assert args.train_model_precision == "bf16"
            assert args.output_dir == "./output"
            assert args.save_steps == 500
            assert args.save_total_limit == 3
            assert args.logging_steps == 1
            assert args.lora_rank == 64
            assert args.lora_alpha == 128
            assert args.lora_dropout == 0.1
            assert args.finetuning_type == "full"
            assert args.training_stage == "sft"
            assert args.ld_alpha == 1.0
            assert args.pref_beta == 0.1
            assert args.dpo_label_smoothing == 0.0
            assert args.sft_weight == 0.0
            
            assert args.trust_remote_code is False
            assert args.tokenizer_use_fast is False
            assert args.enable_gradient_checkpointing is False
            assert args.save_last is False
            assert args.save_train_log is False
            assert args.use_tensorboard is False
            assert args.average_tokens_across_devices is False

    def test_command_line_arguments(self):
        """测试命令行参数解析"""
        test_args = [
            'test_script.py',
            '--model_name_or_path', 'bert-base-uncased',
            '--cutoff_len', '1024',
            '--num_epochs', '5.0',
            '--learning_rate', '1e-4',
            '--train_micro_batch_size_per_gpu', '4',
            '--output_dir', '/tmp/test_output',
            '--training_stage', 'dpo',
        ]
        
        with patch.object(sys, 'argv', test_args):
            args = get_args()
            assert args.model_name_or_path == 'bert-base-uncased'
            assert args.cutoff_len == 1024
            assert args.num_epochs == 5.0
            assert args.learning_rate == 1e-4
            assert args.train_micro_batch_size_per_gpu == 4
            assert args.output_dir == '/tmp/test_output'
            assert args.training_stage == 'dpo'

    def test_boolean_flags(self):
        """测试布尔标志参数"""
        test_args = [
            'test_script.py',
            '--trust_remote_code',
            '--enable_gradient_checkpointing',
            '--save_last',
            '--save_train_log',
            '--use_tensorboard',
            '--average_tokens_across_devices',
        ]
        
        with patch.object(sys, 'argv', test_args):
            args = get_args()
            assert args.trust_remote_code is True
            assert args.enable_gradient_checkpointing is True
            assert args.save_last is True
            assert args.save_train_log is True
            assert args.use_tensorboard is True
            assert args.average_tokens_across_devices is True

    def test_yaml_config_loading(self):
        """测试从 YAML 配置文件加载参数"""
        config_data = {
            'model_name_or_path': 'gpt2',
            'cutoff_len': 2048,
            'num_epochs': 10.0,
            'learning_rate': 5e-5,
            'train_micro_batch_size_per_gpu': 8,
            'output_dir': '/tmp/yaml_output',
            'training_stage': 'pretrain',
            'trust_remote_code': True,
            'enable_gradient_checkpointing': True,
            'lora_rank': 16,
            'lora_alpha': 32,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            test_args = ['test_script.py', '--yaml_config', config_path]
            with patch.object(sys, 'argv', test_args):
                args = get_args()
                assert args.model_name_or_path == 'gpt2'
                assert args.cutoff_len == 2048
                assert args.num_epochs == 10.0
                assert args.learning_rate == 5e-5
                assert args.train_micro_batch_size_per_gpu == 8
                assert args.output_dir == '/tmp/yaml_output'
                assert args.training_stage == 'pretrain'
                assert args.trust_remote_code is True
                assert args.enable_gradient_checkpointing is True
                assert args.lora_rank == 16
                assert args.lora_alpha == 32
        finally:
            os.unlink(config_path)

    def test_command_line_overrides_yaml(self):
        """测试命令行参数覆盖 YAML 配置"""
        config_data = {
            'model_name_or_path': 'gpt2',
            'cutoff_len': 1024,
            'num_epochs': 5.0,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            test_args = [
                'test_script.py',
                '--yaml_config', config_path,
                '--model_name_or_path', 'bert-base-uncased',
                '--cutoff_len', '2048',
            ]
            with patch.object(sys, 'argv', test_args):
                args = get_args()
                assert args.model_name_or_path == 'bert-base-uncased'
                assert args.cutoff_len == 2048
                assert args.num_epochs == 5.0
        finally:
            os.unlink(config_path)

    def test_invalid_yaml_file(self):
        """测试无效的 YAML 文件路径（应该使用默认值）"""
        test_args = ['test_script.py', '--yaml_config', '/nonexistent/path/config.yaml']
        with patch.object(sys, 'argv', test_args):
            args = get_args()
            assert args.cutoff_len == 1024
            assert args.num_epochs == 3.0
            assert args.training_stage == 'sft'

    def test_empty_yaml_file(self):
        """测试空的 YAML 文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('')
            config_path = f.name
        
        try:
            test_args = ['test_script.py', '--yaml_config', config_path]
            with patch.object(sys, 'argv', test_args):
                args = get_args()
                assert args.cutoff_len == 1024
                assert args.training_stage == 'sft'
        finally:
            os.unlink(config_path)

    def test_training_stage_choices(self):
        """测试 training_stage 的 choices 验证"""
        valid_stages = ['pretrain', 'continue_pretrain', 'sft', 'dpo']
        
        for stage in valid_stages:
            test_args = ['test_script.py', '--training_stage', stage]
            with patch.object(sys, 'argv', test_args):
                args = get_args()
                assert args.training_stage == stage

    def test_training_stage_invalid_choice(self):
        """测试无效的 training_stage 值"""
        test_args = ['test_script.py', '--training_stage', 'invalid_stage']
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit):
                get_args()

    def test_train_model_precision_choices(self):
        """测试 train_model_precision 的 choices 验证"""
        valid_precisions = ['fp32', 'fp16', 'bf16']
        
        for precision in valid_precisions:
            test_args = ['test_script.py', '--train_model_precision', precision]
            with patch.object(sys, 'argv', test_args):
                args = get_args()
                assert args.train_model_precision == precision

    def test_train_model_precision_invalid_choice(self):
        """测试无效的 train_model_precision 值"""
        test_args = ['test_script.py', '--train_model_precision', 'invalid']
        with patch.object(sys, 'argv', test_args):
            with pytest.raises(SystemExit):
                get_args()

    def test_lora_target_modules_list(self):
        """测试 lora_target_modules 列表参数"""
        test_args = [
            'test_script.py',
            '--lora_target_modules', 'q_proj', 'k_proj', 'v_proj',
        ]
        with patch.object(sys, 'argv', test_args):
            args = get_args()
            assert args.lora_target_modules == ['q_proj', 'k_proj', 'v_proj']

    def test_lora_target_modules_from_yaml(self):
        """测试从 YAML 加载 lora_target_modules"""
        config_data = {
            'lora_target_modules': ['q_proj', 'k_proj'],
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            test_args = ['test_script.py', '--yaml_config', config_path]
            with patch.object(sys, 'argv', test_args):
                args = get_args()
                assert args.lora_target_modules == ['q_proj', 'k_proj']
        finally:
            os.unlink(config_path)

    def test_float_parameters(self):
        """测试浮点数参数"""
        test_args = [
            'test_script.py',
            '--num_epochs', '2.5',
            '--learning_rate', '1.5e-4',
            '--weight_decay', '0.05',
            '--warmup_ratio', '0.1',
            '--lora_dropout', '0.2',
            '--ld_alpha', '0.5',
            '--pref_beta', '0.2',
            '--dpo_label_smoothing', '0.1',
            '--sft_weight', '0.3',
        ]
        with patch.object(sys, 'argv', test_args):
            args = get_args()
            assert args.num_epochs == 2.5
            assert args.learning_rate == 1.5e-4
            assert args.weight_decay == 0.05
            assert args.warmup_ratio == 0.1
            assert args.lora_dropout == 0.2
            assert args.ld_alpha == 0.5
            assert args.pref_beta == 0.2
            assert args.dpo_label_smoothing == 0.1
            assert args.sft_weight == 0.3

    def test_integer_parameters(self):
        """测试整数参数"""
        test_args = [
            'test_script.py',
            '--cutoff_len', '1024',
            '--train_micro_batch_size_per_gpu', '8',
            '--gradient_accumulation_steps', '4',
            '--save_steps', '1000',
            '--save_total_limit', '5',
            '--logging_steps', '50',
            '--lora_rank', '32',
            '--lora_alpha', '64',
            '--local_rank', '1',
        ]
        with patch.object(sys, 'argv', test_args):
            args = get_args()
            assert args.cutoff_len == 1024
            assert args.train_micro_batch_size_per_gpu == 8
            assert args.gradient_accumulation_steps == 4
            assert args.save_steps == 1000
            assert args.save_total_limit == 5
            assert args.logging_steps == 50
            assert args.lora_rank == 32
            assert args.lora_alpha == 64
            assert args.local_rank == 1

    def test_string_parameters(self):
        """测试字符串参数"""
        test_args = [
            'test_script.py',
            '--model_name_or_path', 'bert-base-uncased',
            '--dataset_path', '/path/to/dataset',
            '--lr_scheduler_type', 'linear',
            '--deepspeed_config_path', '/path/to/deepspeed.json',
            '--output_dir', '/path/to/output',
            '--finetuning_type', 'lora',
        ]
        with patch.object(sys, 'argv', test_args):
            args = get_args()
            assert args.model_name_or_path == 'bert-base-uncased'
            assert args.dataset_path == '/path/to/dataset'
            assert args.lr_scheduler_type == 'linear'
            assert args.deepspeed_config_path == '/path/to/deepspeed.json'
            assert args.output_dir == '/path/to/output'
            assert args.finetuning_type == 'lora'

    def test_optional_parameters_none(self):
        """测试可选参数为 None 的情况"""
        test_args = ['test_script.py']
        with patch.object(sys, 'argv', test_args):
            args = get_args()
            assert args.model_name_or_path is None
            assert args.dataset_path is None
            assert args.deepspeed_config_path is None
            assert args.num_local_io_workers is None
            assert args.lora_target_modules is None

    def test_yaml_with_none_values(self):
        """测试 YAML 中包含 None 值的情况"""
        config_data = {
            'model_name_or_path': None,
            'dataset_path': None,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            test_args = ['test_script.py', '--yaml_config', config_path]
            with patch.object(sys, 'argv', test_args):
                args = get_args()
                assert args.model_name_or_path is None
                assert args.dataset_path is None
        finally:
            os.unlink(config_path)

