import confuse

class LocalConfiguration(confuse.Configuration):
    def config_dir (self):
        return "./"

Config = LocalConfiguration("Pigeonator")