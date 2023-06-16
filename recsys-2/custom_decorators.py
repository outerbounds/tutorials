"""

Custom decorators for Metaflow

"""
from functools import wraps

my_magic_dir = 'merlin'

def magicdir(f):
    artifact = 'magicdir'
    @wraps(f)
    def func(self):
        from io import BytesIO
        from tarfile import TarFile
        existing = getattr(self, artifact, None)
        if existing:
            buf = BytesIO(existing)
            with TarFile(mode='r', fileobj=buf) as tar:
                tar.extractall()
        f(self)
        buf = BytesIO()
        with TarFile(mode='w', fileobj=buf) as tar:
            tar.add(my_magic_dir)
        setattr(self, artifact, buf.getvalue())
    return func
    