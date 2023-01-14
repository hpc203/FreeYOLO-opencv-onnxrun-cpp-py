import argparse
import cv2
import numpy as np
import os

class FreeYOLO():
    def __init__(self, model_path, confThreshold=0.4, nmsThreshold=0.85, datatype='coco'):
        self.net = cv2.dnn.readNet(model_path)
        filename = os.path.splitext(os.path.basename(model_path))[0]
        input_shape = filename.split('_')[-1].split('x')
        self.input_height = int(input_shape[0])
        self.input_width = int(input_shape[1])
        self.anchors, self.expand_strides = self.generate_anchors((self.input_height, self.input_width), [8, 16, 32])

        if datatype=='coco':
            self.classes = list(map(lambda x: x.strip(), open('coco.names', 'r').readlines()))
        elif datatype=='face':
            self.classes = ['face']
        else:
            self.classes = ['person']
        self.num_class = len(self.classes)
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.output_names = self.net.getUnconnectedOutLayersNames()

    def generate_anchors(self, input_shape, strides):
        """
            fmp_size: (List) [H, W]
        """
        all_anchors = []
        all_expand_strides = []
        for stride in strides:
            # generate grid cells
            fmp_h, fmp_w = input_shape[0] // stride, input_shape[1] // stride
            anchor_x, anchor_y = np.meshgrid(np.arange(fmp_w),
                                             np.arange(fmp_h))
            # [H, W, 2]
            anchor_xy = np.stack([anchor_x, anchor_y], axis=-1)
            shape = anchor_xy.shape[:2]
            # [H, W, 2] -> [HW, 2]
            anchor_xy = (anchor_xy.reshape(-1, 2) + 0.5) * stride
            all_anchors.append(anchor_xy)

            # expanded stride
            strides = np.full((*shape, 1), stride)
            all_expand_strides.append(strides.reshape(-1, 1))

        anchors = np.concatenate(all_anchors, axis=0)
        expand_strides = np.concatenate(all_expand_strides, axis=0)

        return anchors, expand_strides

    def decode_boxes(self, anchors, pred_regs, expand_strides):
        """
            anchors:  (List[Tensor]) [1, M, 2] or [M, 2]
            pred_reg: (List[Tensor]) [B, M, 4] or [B, M, 4]
        """
        # center of bbox
        pred_ctr_xy = anchors[..., :2] + pred_regs[..., :2] * expand_strides
        # size of bbox
        pred_box_wh = np.exp(pred_regs[..., 2:]) * expand_strides

        pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
        # pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
        # pred_box = np.concatenate([pred_x1y1, pred_x2y2], axis=-1)
        pred_box = np.concatenate([pred_x1y1, pred_box_wh], axis=-1)
        return pred_box
    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)

        label = '%.2f' % conf
        label = '%s:%s' % (self.classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine), (255,255,255), cv.FILLED)
        cv2.putText(frame, label, (left, top - 10), 0, 0.7, (0, 255, 0), thickness=2)
        return frame

    def detect(self, frame):
        padded_image = np.ones((self.input_height, self.input_width, 3), dtype=np.uint8)*114
        ratio = min(self.input_height / frame.shape[0], self.input_width / frame.shape[1])
        neww, newh = int(frame.shape[1] * ratio), int(frame.shape[0] * ratio)
        temp_image = cv2.resize(frame, (neww, newh), interpolation=cv2.INTER_LINEAR)
        padded_image[:newh, :neww, :] = temp_image
        blob = cv2.dnn.blobFromImage(padded_image)
        self.net.setInput(blob)
        results = self.net.forward(self.output_names)

        reg_preds = results[0][0][..., :4]
        obj_preds = results[0][0][..., 4:5]
        cls_preds = results[0][0][..., 5:]
        scores = np.sqrt(obj_preds * cls_preds)

        # scores & class_ids
        class_ids = np.argmax(scores, axis=1)  # [M,]
        scores = np.max(scores, axis=1)

        # bboxes
        bboxes = self.decode_boxes(self.anchors, reg_preds, self.expand_strides)  # [M, 4]
        # thresh
        keep = np.where(scores > self.confThreshold)
        scores = scores[keep]
        class_ids = class_ids[keep]
        bboxes = bboxes[keep]
        bboxes /= ratio

        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), self.confThreshold, self.nmsThreshold)
        for i in indices:
            left, top, width, height = bboxes[i, :].astype(np.int32)
            frame = self.drawPred(frame, class_ids[i], scores[i], left, top, left + width, top + height)
        return frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelpath", type=str, default='weights/coco/yolo_free_nano_192x320.onnx', help="model path")
    parser.add_argument("--imgpath", type=str, default='images/coco/dog.jpg', help="image path")
    parser.add_argument("--confThreshold", default=0.6, type=float, help='class confidence')
    parser.add_argument("--nmsThreshold", default=0.5, type=float, help='iou thresh')
    parser.add_argument("--datatype", default='coco', type=str, choices=['coco', 'face', 'person'], help='data type')
    args = parser.parse_args()

    net = FreeYOLO(args.modelpath, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold, datatype=args.datatype)
    srcimg = cv2.imread(args.imgpath)
    srcimg = net.detect(srcimg)

    winName = 'Deep learning object detection in OpenCV'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
