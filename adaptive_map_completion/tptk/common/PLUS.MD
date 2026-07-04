使用转换为qgis能够可视化的样子

```shell
utils/tptk/common/hhy_txt_to_csv.py
```

发现无法直接运行，根本原因在于使用了类似“from .trajectory import balabala”的表述，相对引用导致混乱

所以从根目录运行，使用-m确保作为一个模块，需要执行

```shell
(SAM) E:\School\2025\20251022RSTraj>python -m utils.tptk.common.hhy_txt_to_csv

```

然后转化合并为一个大的csv

```shell
utils/tptk/common/hhy_merge_csv.py

```

```shell
正在读取: didi_gaia_xian_2018093004.csv
...
正在读取: didi_gaia_xian_2018100123.csv

合并完成！
生成的总文件: E:\School\2025\20251022RSTraj\dataset\clean\vis\didi_gaia\xian_all_merged.csv
总计合并轨迹数: 148747

```

补充实现把path转为csv的函数

```shell
utils/tptk/common/hhy_path_txt_to_csv.py

```

```shell

```
