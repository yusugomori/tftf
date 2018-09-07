from setuptools import setup
from setuptools import find_packages

setup(
    name='tftf',
    version='0.0.23',
    description='TensorFlow TransFormer',
    author='Yusuke Sugomori',
    author_email='me@yusugomori.com',
    url='https://github.com/yusugomori/tftf',
    download_url='',
    license='Apache 2.0',
    install_requires=['numpy>=1.13.3',
                      'scikit-learn>=0.19.1'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='tensorflow keras machine deep learning',
    packages=find_packages()
)
