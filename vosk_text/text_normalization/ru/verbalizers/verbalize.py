# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from vosk_text.text_normalization.en.graph_utils import GraphFst
from vosk_text.text_normalization.en.verbalizers.whitelist import WhiteListFst
from vosk_text.text_normalization.ru.verbalizers.cardinal import CardinalFst
from vosk_text.text_normalization.ru.verbalizers.date import DateFst
from vosk_text.text_normalization.ru.verbalizers.range import RangeFst
from vosk_text.text_normalization.ru.verbalizers.decimal import DecimalFst
from vosk_text.text_normalization.ru.verbalizers.electronic import ElectronicFst
from vosk_text.text_normalization.ru.verbalizers.measure import MeasureFst
from vosk_text.text_normalization.ru.verbalizers.money import MoneyFst
from vosk_text.text_normalization.ru.verbalizers.ordinal import OrdinalFst
from vosk_text.text_normalization.ru.verbalizers.telephone import TelephoneFst
from vosk_text.text_normalization.ru.verbalizers.time import TimeFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File. 
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize", kind="verbalize", deterministic=deterministic)
        cardinal = CardinalFst()
        cardinal_graph = cardinal.fst
        ordinal_graph = OrdinalFst().fst
        decimal = DecimalFst()
        decimal_graph = decimal.fst
        date = DateFst()
        date_graph = date.fst
        range = RangeFst()
        range_graph = range.fst
        measure = MeasureFst()
        measure_graph = measure.fst
        electronic = ElectronicFst()
        electronic_graph = electronic.fst
        whitelist_graph = WhiteListFst().fst
        money_graph = MoneyFst().fst
        telephone_graph = TelephoneFst().fst
        time_graph = TimeFst().fst

        graph = (
            measure_graph
            | cardinal_graph
            | decimal_graph
            | ordinal_graph
            | date_graph
            | range_graph
            | electronic_graph
            | money_graph
            | whitelist_graph
            | telephone_graph
            | time_graph
        )
        self.fst = graph
