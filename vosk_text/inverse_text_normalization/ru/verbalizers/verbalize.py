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

from vosk_text.inverse_text_normalization.en.verbalizers.whitelist import WhiteListFst
from vosk_text.inverse_text_normalization.ru.verbalizers.cardinal import CardinalFst
from vosk_text.inverse_text_normalization.ru.verbalizers.date import DateFst
from vosk_text.inverse_text_normalization.ru.verbalizers.range import RangeFst
from vosk_text.inverse_text_normalization.ru.verbalizers.decimal import DecimalFst
from vosk_text.inverse_text_normalization.ru.verbalizers.electronic import ElectronicFst
from vosk_text.inverse_text_normalization.ru.verbalizers.measure import MeasureFst
from vosk_text.inverse_text_normalization.ru.verbalizers.money import MoneyFst
from vosk_text.inverse_text_normalization.ru.verbalizers.ordinal import OrdinalFst
from vosk_text.inverse_text_normalization.ru.verbalizers.telephone import TelephoneFst
from vosk_text.inverse_text_normalization.ru.verbalizers.time import TimeFst
from vosk_text.text_normalization.en.graph_utils import GraphFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File. 
    More details to deployment at NeMo/tools/text_processing_deployment.
    """

    def __init__(self):
        super().__init__(name="verbalize", kind="verbalize")
        cardinal = CardinalFst()
        cardinal_graph = cardinal.fst
        ordinal = OrdinalFst()
        ordinal_graph = ordinal.fst
        decimal = DecimalFst()
        decimal_graph = decimal.fst
        whitelist_graph = WhiteListFst().fst
        electronic_graph = ElectronicFst().fst
        money_graph = MoneyFst().fst
        date_graph = DateFst().fst
        date_graph = DateFst().fst
        range_graph = RangeFst().fst
        measure_graph = MeasureFst().fst
        telephone_graph = TelephoneFst().fst
        time_graph = TimeFst().fst

        graph = (
            whitelist_graph
            | cardinal_graph
            | ordinal_graph
            | decimal_graph
            | electronic_graph
            | date_graph
            | range_graph
            | money_graph
            | measure_graph
            | telephone_graph
            | time_graph
        )

        self.fst = graph
