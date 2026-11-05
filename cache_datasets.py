import datasets
for path, name, split in [("facebook/belebele","swe_Latn","test"),("Davlan/sib200","swe_Latn",None),("occiglot/arcX","sv",None),
                          ("LumiOpen/opengpt-x_goldenswagx","SV",None),("LumiOpen/opengpt-x_truthfulqax","mc_SV",None)]:
    d = datasets.load_dataset(path, name)
    print(path, name, {k: len(v) for k, v in d.items()})
