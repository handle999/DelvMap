import os
import glob
import argparse
from tqdm import tqdm
from osgeo import ogr

def get_active_eids(path_folder):
    print(f"[INFO] Scanning {path_folder} for active edge IDs...")
    active_eids = set()
    txt_files = glob.glob(os.path.join(path_folder, "*.txt"))
    
    for fpath in tqdm(txt_files, desc="Parsing Paths"):
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                attrs = line.strip().split(',')
                if len(attrs) >= 3:
                    active_eids.add(int(attrs[2]))
                    
    print(f"[INFO] Found {len(active_eids)} unique active eids.")
    return active_eids

def extract_shp_subset(input_shp, output_shp, active_eids):
    # ==========================================
    # [新增] 自动创建输出文件所在的目录
    # ==========================================
    out_dir = os.path.dirname(output_shp)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        print(f"[INFO] Created output directory: {out_dir}")

    print(f"[INFO] Extracting subset from {input_shp}...")
    
    driver = ogr.GetDriverByName("ESRI Shapefile")
    in_ds = driver.Open(input_shp, 0)
    if in_ds is None:
        raise FileNotFoundError(f"Cannot open input Shapefile: {input_shp}")
        
    in_layer = in_ds.GetLayer()
    
    # 创建输出 SHP
    if os.path.exists(output_shp):
        driver.DeleteDataSource(output_shp)
    out_ds = driver.CreateDataSource(output_shp)
    out_layer = out_ds.CreateLayer("active_edges", geom_type=in_layer.GetGeomType())
    
    # 复制原始字段定义
    in_layer_defn = in_layer.GetLayerDefn()
    for i in range(in_layer_defn.GetFieldCount()):
        out_layer.CreateField(in_layer_defn.GetFieldDefn(i))
        
    out_layer_defn = out_layer.GetLayerDefn()
    
    # 遍历原图，过滤出在 active_eids 中的边
    count = 0
    in_layer.ResetReading()
    for feature in tqdm(in_layer, desc="Filtering SHP", total=in_layer.GetFeatureCount()):
        eid = feature.GetField("eid")
        if eid in active_eids:
            # 复制 Feature
            out_feature = ogr.Feature(out_layer_defn)
            for i in range(out_feature.GetFieldCount()):
                out_feature.SetField(out_layer_defn.GetFieldDefn(i).GetNameRef(), feature.GetField(i))
            
            geom = feature.GetGeometryRef()
            out_feature.SetGeometry(geom.Clone())
            out_layer.CreateFeature(out_feature)
            count += 1
            out_feature = None
            
    print(f"[INFO] Successfully saved {count} features to {output_shp}.")
    
    in_ds = None
    out_ds = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_dir", required=True, help="Dir containing path .txt files")
    parser.add_argument("--in_shp", required=True, help="Original road network SHP")
    parser.add_argument("--out_shp", required=True, help="Output subset SHP")
    args = parser.parse_args()
    
    eids = get_active_eids(args.path_dir)
    extract_shp_subset(args.in_shp, args.out_shp, eids)
    