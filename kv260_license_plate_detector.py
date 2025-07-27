#!/usr/bin/env python3
# kv260_yolov7tiny_test_resize_expanded_ocr.py
import os
import sys
import cv2
import math
import time
import xir
import vart
import torch
import numpy as np
from datetime import datetime
from torchvision.ops import nms

MODEL_PATH       = "dt_model.xmodel"
DATASET_DIR      = "thai_license_plate_dataset_for_yolov7"
TEST_IMAGES_DIR  = os.path.join(DATASET_DIR, "images", "test")
TEST_LABELS_DIR  = os.path.join(DATASET_DIR, "labels", "test")

INPUT_WIDTH, INPUT_HEIGHT = 320, 320
ANCHORS = np.array([[10.0,13.0],[16.0,30.0],[33.0,23.0]])
CLASS_NAMES = ["license_plate"]
CONF_THRESHOLD = 0.10
IOU_THRESHOLD  = 0.30
MAX_BOX_LIMIT  = 20000

# สัดส่วนขยายบ็อกซ์เพื่อส่งเข้า OCR
BOX_EXPAND_RATIO = 0.1  # ขยาย 10% (กว้าง+สูง)

def get_child_subgraph_dpu(graph):
    root_subgraph = graph.get_root_subgraph()
    subgraphs = []
    for sub in root_subgraph.toposort_child_subgraph():
        if sub.has_attr("device") and sub.get_attr("device").upper()=="DPU":
            subgraphs.append(sub)
    return subgraphs

def preprocess_fn(image_bgr):
    """
    Resize (ไม่ Letterbox) => (320x320) แล้ว normalize เป็น int8
    """
    h0, w0 = image_bgr.shape[:2]
    # BGR->RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Resize ปกติ
    image_resized = cv2.resize(image_rgb, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    # Normalize [-1..1]
    image_resized = image_resized.astype(np.float32)/255.0
    image_resized = (image_resized - 0.5)/0.5
    # int8
    image_resized = (image_resized*127.5).astype(np.int8)
    # (H,W,C)->(N,C,H,W)
    pre_data = image_resized.transpose(2,0,1)[np.newaxis,:,:,:]
    return pre_data, w0, h0

def run_dpu(dpu, input_data):
    inTensors  = dpu.get_input_tensors()
    outTensors = dpu.get_output_tensors()
    in_shape = tuple(inTensors[0].dims)
    out_shapes = [tuple(o.dims) for o in outTensors]

    inputData  = [np.empty(in_shape,dtype=np.int8,order='C')]
    outputData = [np.empty(o_shape,dtype=np.int8,order='C') for o_shape in out_shapes]

    inputData[0] = input_data
    job_id = dpu.execute_async(inputData, outputData)
    dpu.wait(job_id)

    outs_float = []
    for i,outT in enumerate(outTensors):
        fix_point = outT.get_attr("fix_point")
        scale_val = 2.0**(-fix_point)
        raw_out = outputData[i]
        if len(raw_out.shape)==4 and raw_out.shape[3]>1:
            raw_out = np.transpose(raw_out,(0,3,1,2))
        out_float = raw_out.astype(np.float32)*scale_val
        outs_float.append(out_float)
    return outs_float

def sigmoid(x):
    return 1.0/(1.0+math.exp(-float(x)))

def decode_predictions(output, anchors, input_w=320, input_h=320, conf_thr=0.10):
    batch_size, channels, grid_h, grid_w = output.shape
    num_anchors = channels//5
    boxes, scores, class_ids = [],[],[]

    stride_x = input_w/float(grid_w)
    stride_y = input_h/float(grid_h)

    for b in range(batch_size):
        data = output[b].reshape(num_anchors,5,grid_h,grid_w)
        for a in range(num_anchors):
            aw, ah = anchors[a]
            for gy in range(grid_h):
                for gx in range(grid_w):
                    tx = data[a,0,gy,gx]
                    ty = data[a,1,gy,gx]
                    tw = data[a,2,gy,gx]
                    th = data[a,3,gy,gx]
                    tc = data[a,4,gy,gx]

                    bx = (sigmoid(tx)+gx)*stride_x
                    by = (sigmoid(ty)+gy)*stride_y
                    bw = aw*math.exp(float(tw))*stride_x
                    bh = ah*math.exp(float(th))*stride_y
                    conf = sigmoid(tc)

                    if conf>=conf_thr:
                        x1 = bx - bw*0.5
                        y1 = by - bh*0.5
                        x2 = bx + bw*0.5
                        y2 = by + bh*0.5
                        if (x2-x1)>1 and (y2-y1)>1:
                            boxes.append([x1,y1,x2,y2])
                            scores.append(conf)
                            class_ids.append(0)
                            if len(boxes)>MAX_BOX_LIMIT:
                                break
                if len(boxes)>MAX_BOX_LIMIT:
                    break
            if len(boxes)>MAX_BOX_LIMIT:
                break
    return boxes,scores,class_ids

def load_ground_truth(label_path, w0, h0):
    boxes=[]
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path,'r') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            cid_str,xc_str,yc_str,ww_str,hh_str=line.split()
            xc=float(xc_str)*w0
            yc=float(yc_str)*h0
            ww=float(ww_str)*w0
            hh=float(hh_str)*h0
            x1 = xc-ww*0.5
            y1 = yc-hh*0.5
            x2 = xc+ww*0.5
            y2 = yc+hh*0.5
            x1=max(0,x1); y1=max(0,y1)
            x2=min(w0-1,x2); y2=min(h0-1,y2)
            if x2>x1 and y2>y1:
                boxes.append([x1,y1,x2,y2])
    return boxes

def box_iou(b1,b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    w = ix2-ix1
    h = iy2-iy1
    if w<=0 or h<=0:
        return 0.0
    inter = w*h
    area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = area1+area2-inter
    if union<=0: return 0.0
    return inter/union

def compute_ap_50(all_dets, all_gts):
    # เรียงตาม conf สูง->ต่ำ
    all_dets = sorted(all_dets, key=lambda x:x[1], reverse=True)
    total_gt=0
    for img_id in all_gts:
        total_gt+=len(all_gts[img_id])

    tp=[]; fp=[]
    matched={}
    for det in all_dets:
        img_id, conf, box = det
        gt_boxes = all_gts.get(img_id,[])
        best_iou=0.0
        best_gt_idx=-1
        for gi,gtb in enumerate(gt_boxes):
            if (img_id,gi) in matched: continue
            iou_val = box_iou(box, gtb)
            if iou_val>best_iou:
                best_iou=iou_val
                best_gt_idx=gi
        if best_iou>=0.5 and best_gt_idx>=0:
            tp.append(1); fp.append(0)
            matched[(img_id,best_gt_idx)] = True
        else:
            tp.append(0); fp.append(1)

    tp=np.array(tp); fp=np.array(fp)
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp/(total_gt+1e-16)
    precision = cum_tp/(cum_tp+cum_fp+1e-16)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(mpre.size-1,0,-1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    idxs = np.where(mrec[1:]!=mrec[:-1])[0]
    ap = np.sum((mrec[idxs+1]-mrec[idxs])*mpre[idxs+1])*100.0
    return ap

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"results_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    images_out_dir = os.path.join(result_dir,"images")
    os.makedirs(images_out_dir, exist_ok=True)
    log_file_path = os.path.join(result_dir,"detection.log")

    if not os.path.isfile(MODEL_PATH):
        print(f"Error: model not found => {MODEL_PATH}")
        sys.exit(1)
    try:
        graph = xir.Graph.deserialize(MODEL_PATH)
    except Exception as e:
        print(f"Error: Failed to deserialize => {MODEL_PATH}, {e}")
        sys.exit(1)
    subgraphs = get_child_subgraph_dpu(graph)
    if not subgraphs:
        print("Error: No DPU subgraph.")
        sys.exit(1)
    dpu = vart.Runner.create_runner(subgraphs[0],"run")

    if not os.path.isdir(TEST_IMAGES_DIR):
        print(f"Error: test images folder not found => {TEST_IMAGES_DIR}")
        sys.exit(1)
    test_images = sorted([f for f in os.listdir(TEST_IMAGES_DIR)
                          if f.lower().endswith(('.jpg','.jpeg','.png'))])

    all_gts={}
    all_dets=[]
    total_time =0.0

    with open(log_file_path,'w') as lf:
        lf.write(f"Start detection - Model: {MODEL_PATH}\n")
        lf.write(f"Test images dir: {TEST_IMAGES_DIR}\n")
        lf.write(f"Results dir: {result_dir}\n\n")

    count_img=0
    for img_file in test_images:
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"Warning: fail read => {img_file}")
            continue
        H0,W0 = bgr.shape[:2]

        # load GT
        label_file = os.path.splitext(img_file)[0]+".txt"
        label_path = os.path.join(TEST_LABELS_DIR, label_file)
        gt_boxes = load_ground_truth(label_path, W0, H0)
        image_id = img_file
        all_gts[image_id] = gt_boxes

        # DPU Inference
        pre_data, w0, h0 = preprocess_fn(bgr)
        t0=time.time()
        outs = run_dpu(dpu, pre_data)
        dt=time.time()-t0
        total_time+=dt
        count_img+=1
        fps = 1.0/dt if dt>0 else 0.0

        boxes_all,scores_all,cls_all = decode_predictions(
            outs[0], ANCHORS,
            input_w=INPUT_WIDTH,input_h=INPUT_HEIGHT,
            conf_thr=CONF_THRESHOLD
        )
        # NMS
        final_boxes=[]
        final_scores=[]
        if len(boxes_all)>0:
            b_t = torch.tensor(boxes_all,dtype=torch.float32)
            s_t = torch.tensor(scores_all,dtype=torch.float32)
            keep_idx = nms(b_t, s_t, IOU_THRESHOLD).numpy()
            final_boxes = b_t[keep_idx].numpy()
            final_scores= s_t[keep_idx].numpy()

        # คลายสเกลย้อนกลับ
        sx = float(W0)/float(INPUT_WIDTH)
        sy = float(H0)/float(INPUT_HEIGHT)

        # Add detection to all_dets
        for i,box in enumerate(final_boxes):
            sc = final_scores[i]
            x1= box[0]*sx
            y1= box[1]*sy
            x2= box[2]*sx
            y2= box[3]*sy
            # บันทึกลง all_dets (ใช้คำนวณ AP)
            all_dets.append((image_id, sc, [x1,y1,x2,y2]))

        # วาดกล่อง + ขยายเผื่อ OCR
        disp = bgr.copy()
        for i,box in enumerate(final_boxes):
            sc = final_scores[i]
            x1= box[0]*sx
            y1= box[1]*sy
            x2= box[2]*sx
            y2= box[3]*sy

            # ขยายกล่องเผื่อสำหรับ OCR
            w_box = x2-x1
            h_box = y2-y1
            expand_w = w_box*BOX_EXPAND_RATIO*0.5
            expand_h = h_box*BOX_EXPAND_RATIO*0.5
            x1e = max(0, x1-expand_w)
            y1e = max(0, y1-expand_h)
            x2e = min(W0-1, x2+expand_w)
            y2e = min(H0-1, y2+expand_h)

            # วาดเส้น
            cv2.rectangle(disp,(int(round(x1e)),int(round(y1e))),(int(round(x2e)),int(round(y2e))),(0,255,0),2)
            txt=f"{CLASS_NAMES[0]}:{sc:.2f}"
            cv2.putText(disp, txt, (int(round(x1e)),max(0,int(round(y1e))-5)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

        # ตั้งชื่อไฟล์สำหรับเซฟ
        if len(final_scores)>0:
            max_conf = float(np.max(final_scores))
            out_name = f"{os.path.splitext(img_file)[0]}_detect_{'YES'}_{max_conf:.2f}.jpg"
        else:
            out_name = f"{os.path.splitext(img_file)[0]}_detect_NO_0.00.jpg"
        out_path = os.path.join(images_out_dir,out_name)
        cv2.imwrite(out_path, disp)

        # เขียน log
        with open(log_file_path,'a') as lf:
            lf.write(f"Image: {img_file}, shape=({W0}x{H0}), dt={dt:.4f}s, FPS={fps:.2f}\n")
            lf.write(f"  GroundTruth {len(gt_boxes)} => {gt_boxes}\n")
            if len(final_boxes)==0:
                lf.write("  Detection=0 => No boxes.\n\n")
                continue
            for i,box in enumerate(final_boxes):
                sc = final_scores[i]
                x1= box[0]*sx
                y1= box[1]*sy
                x2= box[2]*sx
                y2= box[3]*sy
                lf.write(f"  DetBox{i+1}: score={sc:.2f}, box=({x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f})\n")
            lf.write("\n")

    # mAP
    mAP_50 = compute_ap_50(all_dets, all_gts)

    # avg fps
    avg_fps = count_img/total_time if total_time>0 else 0.0

    with open(log_file_path,'a') as lf:
        lf.write(f"\nTotal Test Images={count_img}\n")
        lf.write(f"mAP@0.5={mAP_50:.2f}%\n")
        lf.write(f"Average FPS={avg_fps:.2f}\n")

    print(f"\nDone => {result_dir}")
    print(f"Total Test Images= {count_img}")
    print(f"mAP@0.5= {mAP_50:.2f}%")
    print(f"Average FPS= {avg_fps:.2f}")

if __name__=="__main__":
    main()
