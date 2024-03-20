# DNA-LLM
DNA LLM

ToDo:

1. De-duplication

**Update** Minihash deduplication doesnt work. FCGR works better, however the current plan is to deduplicate based on species
**Update 6/02/24** Certain species have a lot diversity. We'll probably have to plot each sequence by species before deduplicating based on said specie. 

2. Pretraining with GPT-NeoX 

**Update** Transformer models are too memory intensive for long sequences. SSMs or wavelet based models may be a better fit.
**Update 6/02/24** See [slides](https://docs.google.com/presentation/d/1_ygnfKfCyEijrYwlbFfaoJ86T7PLmWovUXToiVrmfl8/edit?usp=sharing) on current plans. Experiment results should provide clarity on what our direction should be. 

3. Create a description - sequence dataset. This will allow us to create a chat like model for creating sequences based on description. An SSM model trained on multimodel datasets can then be finetuned on D-S dataset.
4. Model scaling scripts

## Links (warning, most these links will be out of date due to biosafety considerations)
- [HuggingFace Space](https://huggingface.co/spaces/Hack90/virus_explorer)
- [Project proposal](https://github.com/hssn-20/project-proposal-template)
- [Project workbook](https://docs.google.com/spreadsheets/d/15kc9B6E9O3NX73mFRRXoo8AS_IDCGJ48RHb18I63mNk/edit?usp=sharing)
- [Project weekly slides](https://docs.google.com/presentation/d/1_ygnfKfCyEijrYwlbFfaoJ86T7PLmWovUXToiVrmfl8/edit?usp=sharing)
- [Project overview](https://docs.google.com/presentation/d/1VxHHlj-oJJP8QqPrabcQv0-YYwXhQwiZx7HRmBJ3lb4/edit?usp=sharing)
- [Validation Library](https://github.com/hssn-20/dvq)


