# Copyright (c) 2022 Xingchen Song (sxc19@tsinghua.org.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    name="vosk-text",
    version="0.0.1",
    author="Alpha Cephei Inc",
    author_email="contact@alphacephei.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="Vosk Txt Processing, including TN & ITN",
    url="https://github.com/alphacep/vosk-text",
    packages=find_packages(),
    package_data={'': ['*.tsv', '*.txt', '*.far', '*.fst']},
    install_requires=['pynini', 'regex', 'joblib', 'tqdm', 'inflect'],
    tests_require=['pytest'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
