import yaml
import os
from typing import Dict, Any
import logging

class ConfigurationManager:
    """
    Manages loading and validation of configuration files.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load and validate the configuration file.
        
        Returns:
            Dictionary containing configuration parameters
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self._validate_config(config)
            return config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing configuration file: {e}")

    def _validate_config(self, config: Dict[str, Any]):
        """
        Validate configuration structure and required fields.
        
        Args:
            config: Loaded configuration dictionary
            
        Raises:
            ValueError: If required configuration fields are missing
        """
        required_sections = ['environment', 'sarsa', 'logging']

        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
            
        env_config = config.get('environment', {})
        if not env_config.get('name'):
            raise ValueError("Environment name not specified in configuration")
        
        sarsa_config = config.get('sarsa', {})
        required_sarsa_params = [
            'learning_rate', 'gamma', 'epsilon', 'training', 'logging', 'checkpoint', 'memory'
        ]
        for param in required_sarsa_params:
            if param not in sarsa_config:
                raise ValueError(f"Missing required SARSA parameter: {param}")

    def get_config(self) -> Dict[str, Any]:
        """
        Get the loaded configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary containing configuration updates
        """
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)

    def save_config(self, path: str = None):
        """
        Save current configuration to file.
        
        Args:
            path: Path to save configuration. If None, uses original path
        """
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)