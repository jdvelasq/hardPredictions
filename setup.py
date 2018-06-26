from setuptools import setup

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='hardPredictions',
      version='0.0.0',
      description='Time Series Predictions using Python',
      long_description='A library in Python for time series predictions',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Office/Business :: Time Series',
      ],
      keywords='time-series forecast predict',
      url='http://github.com/jdvelasq/hardPredictions',
      author='Juan David Velásquez & María Alejandra Arango',
      author_email='jdvelasq@unal.edu.co',
      license='MIT',
      packages=['hardPredictions'],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
