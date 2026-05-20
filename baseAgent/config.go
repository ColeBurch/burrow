package baseAgent

import (
	"os"
	"path/filepath"
	"strings"
)

const (
	EnvAgentDir          = "AGENT_DIR" // replace with your ENV_AGENT_DIR value
	DefaultConfigDirName = ".burrow"
)

type PackageConfig struct {
	BurrowConfig *BurrowConfig `json:"burrowConfig"`
}

type BurrowConfig struct {
	ConfigDir string `json:"configDir"`
}

type Config struct {
	ConfigDirName string
}

func NewConfig(pkg PackageConfig) Config {
	configDirName := DefaultConfigDirName
	if pkg.BurrowConfig != nil && pkg.BurrowConfig.ConfigDir != "" {
		configDirName = pkg.BurrowConfig.ConfigDir
	}

	return Config{ConfigDirName: configDirName}
}

func DefaultConfig() Config {
	return Config{ConfigDirName: DefaultConfigDirName}
}

func GetAgentDir() (string, error) {
	return DefaultConfig().GetAgentDir()
}

func (c Config) GetAgentDir() (string, error) {
	if envDir := os.Getenv(EnvAgentDir); envDir != "" {
		return ExpandTildePath(envDir)
	}

	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(homeDir, c.ConfigDirName, "agent"), nil
}

func GetSessionsDir() (string, error) {
	return DefaultConfig().GetSessionsDir()
}

func (c Config) GetSessionsDir() (string, error) {
	agentDir, err := c.GetAgentDir()
	if err != nil {
		return "", err
	}

	return filepath.Join(agentDir, "sessions"), nil
}

func ExpandTildePath(path string) (string, error) {
	if path != "~" && !strings.HasPrefix(path, "~/") {
		return path, nil
	}

	homeDir, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	if path == "~" {
		return homeDir, nil
	}

	return filepath.Join(homeDir, path[2:]), nil
}
