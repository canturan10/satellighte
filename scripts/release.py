import satellighte.version as pkg
import fire


def write_version_file(major: int, minor: int, patch: int):

    version = f'__version__ = "{major}.{minor}.{patch}"'
    with open(pkg.__file__, "w") as version_file:
        version_file.write(version)


def inc_patch():
    version = pkg.__version__
    major, minor, patch = version.split(".")
    write_version_file(major, minor, int(patch) + 1)


def inc_minor():
    version = pkg.__version__
    major, minor, _ = version.split(".")
    write_version_file(major, int(minor) + 1, 0)


def inc_major():
    version = pkg.__version__
    major, _, _ = version.split(".")
    write_version_file(int(major) + 1, 0, 0)


if __name__ == "__main__":
    fire.Fire(
        {
            "get_version": pkg.__version__,
            "inc-patch": inc_patch,
            "inc-minor": inc_minor,
            "inc-major": inc_major,
        }
    )
