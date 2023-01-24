# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import pynini
from vosk_text.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    NEMO_SIGMA,
    GraphFst,
    delete_extra_space,
    delete_space,
    insert_space,
)
from vosk_text.text_normalization.ru.alphabet import RU_ALPHA_OR_SPACE
from vosk_text.text_normalization.ru.utils import get_abs_path
from pynini.lib import pynutil


class RangeFst(GraphFst):
    """
    Finite state transducer for classifying date, e.g. 
        "15x53x34" -> tokens { range { value: "пятнадцать на пятьдесят три на тридцать четыре" } }

    Args:
        number_names: number_names for cardinal and ordinal numbers
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, deterministic: bool):
        super().__init__(name="range", kind="classify", deterministic=deterministic)

        separator = pynini.cross("x", " на ")  # between components
        number = cardinal.cardinal_numbers_default

        mm_word_var = ["миллиметр", "миллиметра", "миллиметров"]
        mm_word = pynini.cross("мм", pynini.union(*mm_word_var))

        sm_word_var = ["сантиметр", "сантиметра", "сантиметров"]
        sm_word = pynini.cross("см", pynini.union(*sm_word_var))

        tagger_graph = number + separator + number + separator + (
            number |
            pynutil.add_weight(number + pynini.accep(" ") + (mm_word | sm_word), -10.0)
        )
        tagger_graph = (pynutil.insert("value: \"") + tagger_graph + pynutil.insert("\"")).optimize()

        # verbalizer
        verbalizer_graph = (
            pynutil.delete("value: \"") + pynini.closure(RU_ALPHA_OR_SPACE, 1) + pynutil.delete("\"")
        )
        verbalizer_graph = verbalizer_graph.optimize()

        self.final_graph = (tagger_graph @ verbalizer_graph).optimize()
        self.fst = pynutil.insert("value: \"") + self.final_graph + pynutil.insert("\"")
        self.fst = self.add_tokens(self.fst).optimize()
