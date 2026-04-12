# =================================================================================================
#  Copyright (c) Innovation First 2025. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
# =================================================================================================
""" 
AIM WebSocket API - Settings

Settings class to read and manage the JSON configuration file.
"""
import json
import os

class Settings:
    """
    Settings class to read the JSON configuration file and provide access to the properties.
    """

    def __init__(self):
        """
        Initialize the Settings class and load the settings from the JSON configuration file.
        """
        self.file_path = os.path.join(os.path.dirname(__file__), 'settings.json')
        self.config = self._load_settings()

    def _load_settings(self):
        """
        Load the settings from the JSON configuration file.

        Returns:
            dict: Dictionary containing the configuration settings.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Configuration file not found: {self.file_path}")

        with open(self.file_path, 'r') as file:
            config = json.load(file)
        
        return config

    @property
    def host(self):
        """
        Get the host property from the configuration settings.

        Returns:
            str: Host property value.
        """
        return self.config.get('connection', {}).get('host', 'localhost')