#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul. 20 09:37 a.m. 2024

@author: AI&RD Research Group, Department of Information Engineering, University of Pisa
"""

from setuptools import setup

setup(name="FederatedFuzzyFRT",
      version="0.1",
      description="Increasing trust in AI through Privacy Preservation and Model Explainability: Federated Learning of Fuzzy Regression Trees library.",
      author="AI&RD Research Group",
      author_email="info@ai.dii.unipi.it",
      packages=["FederatedFuzzyFRT", "FederatedFuzzyFRT.utils"],
      install_requires=['pandas==2.2.2', 'scikit-learn==1.5.1', 'numpy==2.0.0', 'simpful==2.12.0'],
      include_package_data=True,
      long_description=open('README.md').read()
)
