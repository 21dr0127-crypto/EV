---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: "As far as \"\"cost issues\"\" I'd emphasize that our current economic model\
    \ doesn't include the externalities for oil based solutions, which helps in making\
    \ them artificially \"\"cheaper\"\" - this is a political problem that is challenging\
    \ but solvable. A better solution with aggressive carbon pricing would still allow\
    \ oil usage in appropriate circumstances but it would be more limited as it would\
    \ no longer be \"\"cheap\"\" \n\nI agree that large scale energy storage does\
    \ kind of suck right now, and I also agree that we need to get back into nuclear,\
    \ starting yesterday. I still think there's a lot of potential in energy storage,\
    \ more investment in storage is necessary to make renewables work"
- text: Transport and Environment researchers warn that lawmakers have been misled
    by optimistic testing data. “Real-world emissions are going up while official
    numbers go down,” said Sofia Navas Gohlke, co-author of the report. The organisation
    estimates that plug-in hybrids emit only nineteen percent less CO2 than traditional
    gasoline and diesel cars, far from the seventy-five percent reduction claimed
    when the technology was first introduced.
- text: Firstly electric vehicles reduces air pollution which has positive impact
    on human respiratory system, besides this it also reduces noise pollution which
    results in positive mental well-being. Secondly like other machines it also has
    negative impact, such as disposal of used batteries may be a challenge, as it
    filled the lands.
- text: We moved here 3 years ago with our Tesla from Vancouver and I’m honestly debating
    selling our beloved 2020 Model 3. We have a home charger but winter time causes
    me so much range anxiety. Cold Lake is a community about 3.5 hours from Edmonton
    and we have 4 (I think?)
- text: This can lower the risk of breathing problems, asthma, and heart diseases
    for people living near busy roads. EVs are also quieter than petrol or diesel
    vehicles, so they reduce noise pollution, which can improve mental health and
    sleep quality in crowded areas. However, EVs are not completely risk-free.
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: Twitter/twhin-bert-base
---

# SetFit with Twitter/twhin-bert-base

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [Twitter/twhin-bert-base](https://huggingface.co/Twitter/twhin-bert-base) as the Sentence Transformer embedding model. A [SetFitHead](huggingface.co/docs/setfit/reference/main#setfit.SetFitHead) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [Twitter/twhin-bert-base](https://huggingface.co/Twitter/twhin-bert-base)
- **Classification head:** a [SetFitHead](huggingface.co/docs/setfit/reference/main#setfit.SetFitHead) instance
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 4 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
|:------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 0     | <ul><li>'In my opinion, electric vehicles (EVs) have a generally positive impact on human health. Since EVs do not produce tailpipe emissions, they help reduce air pollution, especially in cities, which can lower the risk of respiratory and cardiovascular diseases.'</li><li>"I dont yet know how braking regeneration will figure into this - on the Bolt I dont think about it, it has quite high level of regeneration which I think contributes to good in-town consumption numbers so maybe there is an opportunity to tweak something there. I would highly recommend this car, though all told I am not sure it's better for being an in-town car than a Q6 which I think is on the same new platform? but I wanted a car that would be a better road trip car and from what I understand this car places very high as a road tripper due to a good combination of range and charging speed."</li><li>'Improved air quality can lead to fewer cases of asthma, lung irritation, and other pollution-related health issues. EVs also operate more quietly than conventional vehicles, which helps reduce noise pollution, contributing to lower stress levels and better overall well-being. While there are some concerns related to battery production and disposal, these impacts are mostly indirect and can be managed through cleaner manufacturing processes and proper recycling.'</li></ul> |
| 2     | <ul><li>'An ice vehicle was not designed for a large slab style battery pack to ride along the bottom. Things like the transmission, emissions system and possibly a driveshaft will be in the way. As a result, batteries have to be crammed anywhere they can go.'</li><li>"The real concern lies in the overall speed and how the bike is operated. When used responsibly, a Class 3 e-bike is no more dangerous than any other type of bicycle. We don't limit the power of cars on our roads; instead, we enforce speed limits and other safe driving practices."</li><li>'It might improve air quality if large portion of population uses it. Using "an" EV personally won\'t effect your health in any significant way.'</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 3     | <ul><li>'With a sigh, he stepped into the room and approached Olivia’s bed. “Miss Sinclair, the cancer’s at an advanced stage. You need to start chemotherapy immediately.”\n\n\n\nHe paused, then added gently,\n\n\n\n“Mr.'</li><li>'Taking public transportation, walking instead of driving, choosing EVs, recycling, those are all good and will help reduce your personal carbon footprint, but it’s a drop in the bucket. Corporations[ produce 70% of carbon emissions](https://cdn.cdp.net/cdp-production/cms/reports/documents/000/002/327/original/Carbon-Majors-Report-2017.pdf?1501833772), meaning the change needs to happen there. Where the focus needs to be is industrial and large scale reform.'</li><li>'So please tell me exactly what the ""give away horse shit"" is specifically, because this is a great start to bringing healthcare costs down. Because the other major points sound good to me too:  \n  \nClimate and Energy Investments:\n\n* Commits approximately $369 billion to climate and clean energy programs (creating quality jobs, advancing science, and hopefully slowing climmate change.)'</li></ul>                                                                                                                                                                                                                                                              |
| 1     | <ul><li>'Recycling exposure – Workers involved in battery recycling may face higher exposure to heavy metals and toxic dust if protections are inadequate.'</li><li>"If an area uses mostly coal power, using an EV is basically running on coal. What's worse, EVs increase energy demand, but the environmental burden is shifted to poorer areas with power plants in them, while richer areas benefit from fewer emissions from EVs, since poor people tend not to be able to afford EVs. Without better energy infrastructure, electric vehicles are just shifting the consequences to hurt the poor and benefit the rich, while leaving the actual problem largely unaddressed."</li><li>'Overall, I think EVs are **far healthier than traditional vehicles**, particularly for public health and urban living. With proper battery recycling and safety measures, their health benefits clearly outweigh the risks.'</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("We moved here 3 years ago with our Tesla from Vancouver and I’m honestly debating selling our beloved 2020 Model 3. We have a home charger but winter time causes me so much range anxiety. Cold Lake is a community about 3.5 hours from Edmonton and we have 4 (I think?)")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median  | Max |
|:-------------|:----|:--------|:----|
| Word count   | 3   | 52.8462 | 784 |

| Label | Training Sample Count |
|:------|:----------------------|
| 0     | 361                   |
| 1     | 143                   |
| 2     | 145                   |
| 3     | 229                   |

### Training Hyperparameters
- batch_size: (4, 4)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 20
- body_learning_rate: (2e-05, 2e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: True
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0001 | 1    | 0.247         | -               |
| 0.0057 | 50   | 0.3422        | -               |
| 0.0114 | 100  | 0.3161        | 0.3028          |
| 0.0171 | 150  | 0.3066        | -               |
| 0.0228 | 200  | 0.2742        | 0.2666          |
| 0.0285 | 250  | 0.271         | -               |
| 0.0342 | 300  | 0.2486        | 0.2555          |
| 0.0399 | 350  | 0.2289        | -               |
| 0.0456 | 400  | 0.2425        | 0.2445          |
| 0.0513 | 450  | 0.2485        | -               |
| 0.0569 | 500  | 0.2387        | 0.2505          |
| 0.0626 | 550  | 0.2383        | -               |
| 0.0683 | 600  | 0.2545        | 0.2400          |
| 0.0740 | 650  | 0.2275        | -               |
| 0.0797 | 700  | 0.2381        | 0.2796          |
| 0.0854 | 750  | 0.2395        | -               |
| 0.0911 | 800  | 0.2129        | 0.2760          |
| 0.0968 | 850  | 0.23          | -               |
| 0.1025 | 900  | 0.2134        | 0.2565          |

### Framework Versions
- Python: 3.14.0
- SetFit: 1.1.3
- Sentence Transformers: 5.2.0
- Transformers: 4.57.3
- PyTorch: 2.9.1+cu130
- Datasets: 4.4.2
- Tokenizers: 0.22.1

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->
