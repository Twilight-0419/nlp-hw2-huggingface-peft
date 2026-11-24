# 数据准备

按[原始作业说明](https://github.com/WangXiangyv/NLPDL-2026Fall/tree/main/hw2_huggingface)获取监督数据，然后在当前目录下建立以下结构。

```text
supervised/
  restaurant_sup/
    train.json
    test.json
  laptop_sup/
    train.json
    test.json
  acl_sup/
    train.jsonl
    test.jsonl
  agnews_sup/
    agnews_sup.csv
```

Restaurant 和 Laptop 的 JSON 记录使用 `term`、`sentence`、`polarity` 字段。ACL-ARC 每行包含 `text` 和 `label`。AG News CSV 无表头，列顺序为标签、标题、描述。
