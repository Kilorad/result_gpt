# result_gpt
Only-code version, no heavy models and data
Version with heavy models and data here:
https://disk.yandex.ru/d/1n-x0ZBzKkOY3Q
Launch order:
1) Install vizdoom (this should be sufficient https://pypi.org/project/vizdoom/ )
2) Create dataset from parts by collect_dataset.ipynb (not nesessary)
3) Fit model by cross_gpt_pretrain_medium_bigdata_contrast_2.ipynb (not nesessary)
4) Launch gpt in doom by doom_gpt_memory_descr.ipynb

doom_gpt_memory_descr.ipynb does not contain full RL cycle. If you need to learn, you should launch (2) and (3).

You can launch result gpt not in Doom an learn it not in Doom. This requires updating of doom_gpt_memory_descr.ipynb file. You should connect another environment.

You can test model in answer_question.ipynb (use images and generate images too). Be careful, it is not main mode, system is stupid.