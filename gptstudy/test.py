# my api key: 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

import json
import os
import openai

openai.api_key = "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"

models = openai.Model.list()

print(models)
