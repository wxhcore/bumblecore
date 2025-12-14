import pytest
import sys
import tempfile
import yaml
from unittest.mock import patch, MagicMock
from bumblecore.cli.arg_parser import get_args
from bumblecore.config import TrainConfig


class TestLauncherEndToEnd:
    """测试 launcher 模块的端到端功能：从命令行参数到 TrainConfig"""

    def test_launch_train_from_cli_basic(self):
        """测试基本的命令行参数到 TrainConfig 的转换"""
        test_args = [
            'test_script.py',
            '--model_name_or_path', 'bert-base-uncased',
            '--dataset_path', '/path/to/dataset',
            '--output_dir', '/path/to/output',
            '--training_stage', 'sft',
            '--cutoff_len', '1024',
            '--num_epochs', '5.0',
            '--learning_rate', '1e-4',
        ]
        
        with patch.object(sys, 'argv', test_args):
            args = get_args()
            args_dict = vars(args)
            
            args_dict.pop('yaml_config', None)
            args_dict.pop('local_rank', None)
            
            config = TrainConfig(**args_dict)
        
        assert isinstance(config, TrainConfig)
        
        assert config.model_name_or_path == 'bert-base-uncased'
        assert config.dataset_path == '/path/to/dataset'
        assert config.output_dir == '/path/to/output'
        assert config.training_stage == 'sft'
        assert config.cutoff_len == 1024
        assert config.num_epochs == 5.0
        assert config.learning_rate == 1e-4
        
        assert not hasattr(config, 'yaml_config')
        assert not hasattr(config, 'local_rank')

    def test_launch_train_from_cli_with_yaml_config(self):
        """测试从 YAML 配置文件加载参数"""
        config_data = {
            'model_name_or_path': 'gpt2',
            'dataset_path': '/yaml/dataset',
            'output_dir': '/yaml/output',
            'training_stage': 'dpo',
            'cutoff_len': 2048,
            'num_epochs': 10.0,
            'learning_rate': 5e-5,
            'lora_rank': 16,
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            test_args = [
                'test_script.py',
                '--yaml_config', config_path,
            ]
            
            with patch.object(sys, 'argv', test_args):
                args = get_args()
                args_dict = vars(args)
                
                # 模拟 launcher.py 中的处理逻辑
                args_dict.pop('yaml_config', None)
                args_dict.pop('local_rank', None)
                
                config = TrainConfig(**args_dict)
            
            assert config.model_name_or_path == 'gpt2'
            assert config.dataset_path == '/yaml/dataset'
            assert config.output_dir == '/yaml/output'
            assert config.training_stage == 'dpo'
            assert config.cutoff_len == 2048
            assert config.num_epochs == 10.0
            assert config.learning_rate == 5e-5
            assert config.lora_rank == 16
            
            assert not hasattr(config, 'yaml_config')
        finally:
            import os
            os.unlink(config_path)

    def test_launch_train_from_cli_command_line_overrides_yaml(self):
        """测试命令行参数覆盖 YAML 配置"""
        config_data = {
            'model_name_or_path': 'gpt2',
            'dataset_path': '/yaml/dataset',
            'output_dir': '/yaml/output',
            'cutoff_len': 1024,
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
                args_dict = vars(args)
                
                args_dict.pop('yaml_config', None)
                args_dict.pop('local_rank', None)
                
                config = TrainConfig(**args_dict)
            
            assert config.model_name_or_path == 'bert-base-uncased'
            assert config.cutoff_len == 2048
            
            assert config.dataset_path == '/yaml/dataset'
            assert config.output_dir == '/yaml/output'
        finally:
            import os
            os.unlink(config_path)

    def test_launch_train_from_cli_removes_yaml_config_and_local_rank(self):
        """测试 yaml_config 和 local_rank 被正确移除"""
        test_args = [
            'test_script.py',
            '--model_name_or_path', 'test-model',
            '--dataset_path', '/test/dataset',
            '--output_dir', '/test/output',
            '--yaml_config', '/some/config.yaml',
            '--local_rank', '1',
        ]
        
        with patch.object(sys, 'argv', test_args):
            args = get_args()
            args_dict = vars(args)
            
            args_dict.pop('yaml_config', None)
            args_dict.pop('local_rank', None)
            
            config = TrainConfig(**args_dict)
        
        assert not hasattr(config, 'yaml_config')
        assert not hasattr(config, 'local_rank')
        
        assert config.model_name_or_path == 'test-model'
        assert config.dataset_path == '/test/dataset'
        assert config.output_dir == '/test/output'

    def test_launch_train_from_cli_all_parameters(self):
        """测试所有参数都能正确传递到 TrainConfig"""
        test_args = [
            'test_script.py',
            '--model_name_or_path', 'bert-base-uncased',
            '--dataset_path', '/path/to/dataset',
            '--output_dir', '/path/to/output',
            '--trust_remote_code',
            '--cutoff_len', '2048',
            '--num_epochs', '5.0',
            '--learning_rate', '2e-5',
            '--weight_decay', '0.05',
            '--lr_scheduler_type', 'linear',
            '--enable_gradient_checkpointing',
            '--warmup_ratio', '0.1',
            '--train_micro_batch_size_per_gpu', '8',
            '--gradient_accumulation_steps', '4',
            '--train_model_precision', 'fp16',
            '--deepspeed_config_path', '/path/to/deepspeed.json',
            '--num_local_io_workers', '4',
            '--save_steps', '1000',
            '--save_total_limit', '5',
            '--save_last',
            '--logging_steps', '50',
            '--save_train_log',
            '--use_tensorboard',
            '--average_tokens_across_devices',
            '--lora_rank', '32',
            '--lora_alpha', '64',
            '--lora_dropout', '0.2',
            '--lora_target_modules', 'q_proj', 'k_proj', 'v_proj',
            '--finetuning_type', 'lora',
            '--training_stage', 'dpo',
            '--ld_alpha', '0.5',
            '--pref_beta', '0.2',
            '--dpo_label_smoothing', '0.1',
            '--sft_weight', '0.3',
        ]
        
        with patch.object(sys, 'argv', test_args):
            args = get_args()
            args_dict = vars(args)
            
            args_dict.pop('yaml_config', None)
            args_dict.pop('local_rank', None)
            
            config = TrainConfig(**args_dict)
        
        assert config.model_name_or_path == 'bert-base-uncased'
        assert config.dataset_path == '/path/to/dataset'
        assert config.output_dir == '/path/to/output'
        assert config.trust_remote_code is True
        assert config.cutoff_len == 2048
        assert config.num_epochs == 5.0
        assert config.learning_rate == 2e-5
        assert config.weight_decay == 0.05
        assert config.lr_scheduler_type == 'linear'
        assert config.enable_gradient_checkpointing is True
        assert config.warmup_ratio == 0.1
        assert config.train_micro_batch_size_per_gpu == 8
        assert config.gradient_accumulation_steps == 4
        assert config.train_model_precision == 'fp16'
        assert config.deepspeed_config_path == '/path/to/deepspeed.json'
        assert config.num_local_io_workers == 4
        assert config.save_steps == 1000
        assert config.save_total_limit == 5
        assert config.save_last is True
        assert config.logging_steps == 50
        assert config.save_train_log is True
        assert config.use_tensorboard is True
        assert config.average_tokens_across_devices is True
        assert config.lora_rank == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.2
        assert config.lora_target_modules == ['q_proj', 'k_proj', 'v_proj']
        assert config.finetuning_type == 'lora'
        assert config.training_stage == 'dpo'
        assert config.ld_alpha == 0.5
        assert config.pref_beta == 0.2
        assert config.dpo_label_smoothing == 0.1
        assert config.sft_weight == 0.3

    def test_launch_train_from_cli_different_training_stages(self):
        """测试不同训练阶段的配置"""
        training_stages = ['pretrain', 'continue_pretrain', 'sft', 'dpo']
        
        for stage in training_stages:
            test_args = [
                'test_script.py',
                '--model_name_or_path', 'test-model',
                '--dataset_path', '/test/dataset',
                '--output_dir', '/test/output',
                '--training_stage', stage,
            ]
            
            with patch.object(sys, 'argv', test_args):
                args = get_args()
                args_dict = vars(args)
                
                args_dict.pop('yaml_config', None)
                args_dict.pop('local_rank', None)
                
                config = TrainConfig(**args_dict)
            
            assert config.training_stage == stage
            assert isinstance(config, TrainConfig)

    def test_launch_train_from_cli_missing_required_fields(self):
        """测试缺少必需字段时应该抛出 ValueError"""
        test_args = [
            'test_script.py',
        ]
        
        with patch.object(sys, 'argv', test_args):
            args = get_args()
            args_dict = vars(args)
            
            args_dict.pop('yaml_config', None)
            args_dict.pop('local_rank', None)
            
            args_dict['model_name_or_path'] = args_dict.get('model_name_or_path') or ""
            args_dict['dataset_path'] = args_dict.get('dataset_path') or ""
            args_dict['output_dir'] = args_dict.get('output_dir') or ""
            
            with pytest.raises(ValueError, match="must be specified and non-empty"):
                TrainConfig(**args_dict)

    def test_launch_train_from_cli_with_defaults(self):
        """测试使用默认值的情况"""
        test_args = [
            'test_script.py',
            '--model_name_or_path', 'test-model',
            '--dataset_path', '/test/dataset',
            '--output_dir', '/test/output',
        ]
        
        with patch.object(sys, 'argv', test_args):
            args = get_args()
            args_dict = vars(args)
            
            args_dict.pop('yaml_config', None)
            args_dict.pop('local_rank', None)
            
            config = TrainConfig(**args_dict)
        
        assert config.cutoff_len == 1024
        assert config.num_epochs == 3.0
        assert config.learning_rate == 5e-5
        assert config.weight_decay == 0.01
        assert config.lr_scheduler_type == 'cosine'
        assert config.warmup_ratio == 0.1
        assert config.train_micro_batch_size_per_gpu == 4
        assert config.gradient_accumulation_steps == 8
        assert config.train_model_precision == 'bf16'
        assert config.output_dir == '/test/output' 
        assert config.training_stage == 'sft'
        assert config.finetuning_type == 'full'

