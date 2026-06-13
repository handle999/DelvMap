#!/usr/bin/env python3
"""
生成所有epoch的可视化汇总页面
用法: python generate_all_epochs_html.py
"""
import os
import re

# 配置路径
WEB_DIR = "checkpoints/delvmap_exp1/web"
OUTPUT_FILE = os.path.join(WEB_DIR, "all_epochs.html")
IMAGES_DIR = os.path.join(WEB_DIR, "images")

# 获取所有epoch
def get_all_epochs():
    files = os.listdir(IMAGES_DIR)
    epochs = set()
    for f in files:
        match = re.match(r'(epoch\d+)_.*\.png', f)
        if match:
            epochs.add(match.group(1))
    return sorted(epochs, key=lambda x: int(x.replace('epoch', '')))

# 图片类型（与现有index.html一致）
IMAGE_TYPES = [
    ('img_1', 'Traj Input'),
    ('img_2', 'Src Input'),
    ('label', 'GT Road'),
    ('pred_traj_img', 'Pred Road (Traj+Src)'),
    ('src', 'Satellite'),
    ('building_label', 'GT Building'),
    ('pred_building_img', 'Pred Building'),
    ('pred_src_traj_img', 'Pred Road (Src Only)'),
]

def generate_html():
    epochs = get_all_epochs()
    print(f"Found {len(epochs)} epochs")

    html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DelvMap - All Epochs Visualization</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .epoch-row { background: white; margin: 10px 0; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .epoch-header { font-size: 18px; font-weight: bold; color: #2196F3; margin-bottom: 10px; }
        table { table-layout: fixed; width: 100%; }
        td { text-align: center; vertical-align: top; padding: 5px; }
        img { width: 150px; height: 150px; object-fit: contain; border: 1px solid #ddd; }
        img:hover { border-color: #2196F3; }
        .label { font-size: 12px; color: #666; margin-top: 5px; }
        .nav { position: fixed; top: 10px; right: 10px; background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }
        .nav a { margin: 0 5px; text-decoration: none; color: #2196F3; }
        .jump-input { width: 50px; }
    </style>
</head>
<body>
    <h1>DelvMap - All Epochs Visualization</h1>
    <p>Total Epochs: ''' + str(len(epochs)) + ''' | <a href="index.html">Latest Epoch</a></p>

    <div class="nav">
        <a href="#" onclick="scrollToTop()">Top</a> |
        <input type="number" id="jumpEpoch" class="jump-input" min="1" max="''' + str(len(epochs)) + '''">
        <button onclick="jumpToEpoch()">Go</button>
    </div>

'''

    for epoch in epochs:
        epoch_num = int(epoch.replace('epoch', ''))
        html += f'    <div class="epoch-row" id="epoch{epoch_num}">\n'
        html += f'        <div class="epoch-header">Epoch {epoch_num:03d}</div>\n'
        html += '        <table border="1" style="table-layout: fixed;">\n'
        html += '            <tr>\n'

        for img_suffix, label in IMAGE_TYPES:
            img_filename = f"{epoch}_{img_suffix}.png"
            img_path = f"images/{img_filename}"
            if os.path.exists(os.path.join(IMAGES_DIR, img_filename)):
                html += f'''                <td valign="top">
                    <a href="{img_path}">
                        <img src="{img_path}">
                    </a><br>
                    <p class="label">{label}</p>
                </td>
'''
            else:
                html += f'                <td><p class="label">{label} (N/A)</p></td>\n'

        html += '            </tr>\n'
        html += '        </table>\n'
        html += '    </div>\n'

    html += '''    <script>
        function scrollToTop() {
            window.scrollTo(0, 0);
        }

        function jumpToEpoch() {
            var num = document.getElementById('jumpEpoch').value;
            var element = document.getElementById('epoch' + num);
            if (element) {
                element.scrollIntoView({behavior: 'smooth'});
            }
        }
    </script>
</body>
</html>'''

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Generated: {OUTPUT_FILE}")
    print(f"Total epochs: {len(epochs)}")

if __name__ == "__main__":
    generate_html()
