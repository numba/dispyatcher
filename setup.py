from distutils.core import setup
setup(name='dispyatcher',
      version='0.1',
      description='Callsite generation for Python using LLVM',
      author='Andre Masella',
      author_email='amasella@anaconda.com',
      url='https://github.com/numba/dispyatcher',
      packages=['dispyatcher', 'distutils.command'],
      test_suite='tests',
      )
