from distutils.core import setup, Extension

module = Extension("wrapper",
                   sources=[],
                   libraries=["udev", "m"],
                   library_dirs=["lib"])

setup(name="wrapper",
      version="1.0",
      description="Wrapper Extension",
      ext_modules=[module])