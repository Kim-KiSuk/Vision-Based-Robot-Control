import os
import cv2
import json
import argparse
import numpy as np
import pyrealsense2 as rs

# 1. RealSense 사진 2장 촬영
def capture_two_images(out_dir, width=640, height=360, fps=30):

    os.makedirs(out_dir, exist_ok=True)

    img1_path = os.path.join(out_dir, "view1.png")
    img2_path = os.path.join(out_dir, "view2.png")

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    pipeline.start(config)
    count = 0
    paths = [img1_path, img2_path]

    try:
        while count < 2:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            display = img.copy()

            cv2.putText(
                display,
                f"Capture image {count + 1}/2 : press 's'",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("RealSense Capture", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                cv2.imwrite(paths[count], img)
                print(f"Saved: {paths[count]}")
                count += 1

            elif key == ord("q"):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if count < 2:
        raise RuntimeError("Not 2 images")

    return img1_path, img2_path


# 2. Calibration 결과 불러오기
def load_calibration(calib_path, image_shape):
    with open(calib_path, "r") as f:
        data = json.load(f)

    K = np.array(data["camera_matrix"], dtype=np.float64)

    dist = None
    if "distortion_coefficients" in data:
        dist = np.array(data["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)

    img_h, img_w = image_shape[:2]

    calib_w = data.get("image_width", img_w)
    calib_h = data.get("image_height", img_h)
    
    if calib_w != img_w or calib_h != img_h:
        sx = img_w / calib_w
        sy = img_h / calib_h

        K[0, :] *= sx
        K[1, :] *= sy
    return K, dist


# 3. 수동 대응점 클릭
class PointSelector:
    def __init__(self, image, window_name, min_points=8, required_points=None, max_display_width=1200):
        self.original = image.copy()
        self.window_name = window_name
        self.min_points = min_points
        self.required_points = required_points
        self.points = []

        h, w = image.shape[:2]
        if w > max_display_width:
            self.scale = max_display_width / w
        else:
            self.scale = 1.0

        self.display_w = int(w * self.scale)
        self.display_h = int(h * self.scale)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            px = x / self.scale
            py = y / self.scale

            self.points.append([px, py])
            print(f"{self.window_name}: point {len(self.points)} = ({px:.2f}, {py:.2f})")

    def draw(self):
        disp = cv2.resize(self.original, (self.display_w, self.display_h))

        for i, p in enumerate(self.points):
            x = int(p[0] * self.scale)
            y = int(p[1] * self.scale)

            cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                disp,
                str(i),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        guide1 = "Left click: add point | z: undo | r: reset | enter: finish | q: quit"
        guide2 = f"points: {len(self.points)}"

        if self.required_points is not None:
            guide2 += f" / required: {self.required_points}"
        else:
            guide2 += f" / minimum: {self.min_points}"

        cv2.putText(disp, guide1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(disp, guide2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        return disp

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            disp = self.draw()
            cv2.imshow(self.window_name, disp)

            key = cv2.waitKey(20) & 0xFF

            if key == ord("z"):
                if len(self.points) > 0:
                    removed = self.points.pop()
                    print("Undo")

            elif key == ord("r"):
                self.points = []
                print("Reset")

            elif key == ord("q"):
                cv2.destroyWindow(self.window_name)
                return None

            elif key == 13 or key == 10:
                if self.required_points is not None:
                    if len(self.points) == self.required_points:
                        break
                    else:
                        print(self.required_points)
                else:
                    if len(self.points) >= self.min_points:
                        break
                    else:
                        print(self.min_points)

        cv2.destroyWindow(self.window_name)
        return np.array(self.points, dtype=np.float64)


def select_corresponding_points(img1, img2, min_points=8):
    selector1 = PointSelector(img1, "Image 1 - select points", min_points=min_points)
    pts1 = selector1.run()

    selector2 = PointSelector(
        img2,
        "Image 2 - select corresponding points",
        min_points=min_points,
        required_points=len(pts1)
    )
    pts2 = selector2.run()

    return pts1, pts2

# 4. Pixel 좌표 -> Normalized camera 좌표
def pixel_to_normalized_points(pts_px, K, dist=None, use_distortion_correction=True):
    pts_px = np.asarray(pts_px, dtype=np.float64).reshape(-1, 1, 2)

    if use_distortion_correction and dist is not None:
        pts_norm = cv2.undistortPoints(pts_px, K, dist)
        pts_norm = pts_norm.reshape(-1, 2)
    else:
        pts = pts_px.reshape(-1, 2)
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        pts_h = np.hstack([pts, ones])

        K_inv = np.linalg.inv(K)
        pts_norm_h = (K_inv @ pts_h.T).T
        pts_norm = pts_norm_h[:, :2] / pts_norm_h[:, 2:3]

    return pts_norm

# 5. Eight-point linear algorithm
def estimate_essential_eight_point(pts1_norm, pts2_norm):
    if pts1_norm.shape[0] < 8:
        return None

    n = pts1_norm.shape[0]
    A = np.zeros((n, 9), dtype=np.float64)

    for i in range(n):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]

        A[i] = [
            x2 * x1,
            x2 * y1,
            x2,
            y2 * x1,
            y2 * y1,
            y2,
            x1,
            y1,
            1.0
        ]

    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(E)

    s = 0.5 * (S[0] + S[1])
    E = U @ np.diag([s, s, 0.0]) @ Vt
    
    E = E / np.linalg.norm(E)

    return E

# 6. Essential matrix -> R, t 후보 분해
def triangulate_points_normalized(pts1_norm, pts2_norm, R, t):
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])

    points_3d = []

    for p1, p2 in zip(pts1_norm, pts2_norm):
        x1, y1 = p1
        x2, y2 = p2

        A = np.zeros((4, 4), dtype=np.float64)

        A[0] = x1 * P1[2] - P1[0]
        A[1] = y1 * P1[2] - P1[1]
        A[2] = x2 * P2[2] - P2[0]
        A[3] = y2 * P2[2] - P2[1]

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]

        points_3d.append(X[:3])

    return np.array(points_3d)


def count_positive_depth(points_3d, R, t):
    count = 0

    for X1 in points_3d:
        depth1 = X1[2]

        X2 = R @ X1 + t
        depth2 = X2[2]

        if depth1 > 0 and depth2 > 0:
            count += 1

    return count


def decompose_essential_and_choose_pose(E, pts1_norm, pts2_norm):
    U, S, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1

    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 1]
    ], dtype=np.float64)

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    if np.linalg.det(R1) < 0:
        R1 = -R1

    if np.linalg.det(R2) < 0:
        R2 = -R2

    t = U[:, 2]

    candidates = [
        (R1,  t),
        (R1, -t),
        (R2,  t),
        (R2, -t)
    ]

    best = None
    best_count = -1
    best_points_3d = None

    for R_candidate, t_candidate in candidates:
        points_3d = triangulate_points_normalized(
            pts1_norm,
            pts2_norm,
            R_candidate,
            t_candidate
        )

        positive_count = count_positive_depth(points_3d, R_candidate, t_candidate)

        if positive_count > best_count:
            best_count = positive_count
            best = (R_candidate, t_candidate)
            best_points_3d = points_3d

    R_best, t_best = best
    t_best = t_best / np.linalg.norm(t_best)

    return R_best, t_best, best_count, best_points_3d

# 7. Error 계산
def compute_epipolar_errors(E, pts1_norm, pts2_norm):
    errors = []

    for p1, p2 in zip(pts1_norm, pts2_norm):
        x1 = np.array([p1[0], p1[1], 1.0])
        x2 = np.array([p2[0], p2[1], 1.0])

        value = x2.T @ E @ x1

        Ex1 = E @ x1
        Etx2 = E.T @ x2

        denom = Ex1[0] ** 2 + Ex1[1] ** 2 + Etx2[0] ** 2 + Etx2[1] ** 2

        if denom < 1e-12:
            errors.append(0.0)
        else:
            sampson_error = (value ** 2) / denom
            errors.append(np.sqrt(sampson_error))

    return np.array(errors)


# 8. 결과 시각화 저장
def save_match_visualization(img1, img2, pts1, pts2, out_path):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    target_h = min(h1, h2)

    scale1 = target_h / h1
    scale2 = target_h / h2

    img1_r = cv2.resize(img1, (int(w1 * scale1), target_h))
    img2_r = cv2.resize(img2, (int(w2 * scale2), target_h))

    canvas = np.hstack([img1_r, img2_r])
    offset_x = img1_r.shape[1]

    for i, (p1, p2) in enumerate(zip(pts1, pts2)):
        x1 = int(p1[0] * scale1)
        y1 = int(p1[1] * scale1)

        x2 = int(p2[0] * scale2) + offset_x
        y2 = int(p2[1] * scale2)

        cv2.circle(canvas, (x1, y1), 5, (0, 0, 255), -1)
        cv2.circle(canvas, (x2, y2), 5, (0, 0, 255), -1)

        cv2.line(canvas, (x1, y1), (x2, y2), (255, 0, 0), 1)

        cv2.putText(canvas, str(i), (x1 + 6, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(canvas, str(i), (x2 + 6, y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    cv2.imwrite(out_path, canvas)


# 9. Main
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--calib", type=str, default="calibration_result/camera_calibration.json")
    parser.add_argument("--img1", type=str, default=None)
    parser.add_argument("--img2", type=str, default=None)

    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--outdir", type=str, default="eight_point_result")

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)

    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--no-distortion-correction", action="store_true")


    parser.add_argument("--scale-pair", nargs=2, type=int, default=None)
    parser.add_argument("--scale-distance", type=float, default=None)

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.capture:
        img1_path, img2_path = capture_two_images(
            args.outdir,
            width=args.width,
            height=args.height,
            fps=args.fps
        )

        img1_path = args.img1
        img2_path = args.img2

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    K, dist = load_calibration(args.calib, img1.shape)

    if dist is not None:
        print("\nDistortion coefficients:")
        print(dist.reshape(-1))

    pts1_px, pts2_px = select_corresponding_points(
        img1,
        img2,
        min_points=args.min_points
    )

    use_distortion_correction = not args.no_distortion_correction

    pts1_norm = pixel_to_normalized_points(
        pts1_px,
        K,
        dist,
        use_distortion_correction=use_distortion_correction
    )

    pts2_norm = pixel_to_normalized_points(
        pts2_px,
        K,
        dist,
        use_distortion_correction=use_distortion_correction
    )

    E = estimate_essential_eight_point(pts1_norm, pts2_norm)

    R, t, positive_depth_count, points_3d = decompose_essential_and_choose_pose(
        E,
        pts1_norm,
        pts2_norm
    )

    epipolar_errors = compute_epipolar_errors(E, pts1_norm, pts2_norm)

    t_scaled = None
    scale_factor = None

    if args.scale_pair is not None and args.scale_distance is not None:
        i, j = args.scale_pair


        reconstructed_distance = np.linalg.norm(points_3d[i] - points_3d[j])


        scale_factor = args.scale_distance / reconstructed_distance
        t_scaled = t * scale_factor

    # 결과 저장
    result = {
        "num_points": int(len(pts1_px)),
        "K": K.tolist(),
        "distortion_coefficients": None if dist is None else dist.reshape(-1).tolist(),
        "points_image1_pixel": pts1_px.tolist(),
        "points_image2_pixel": pts2_px.tolist(),
        "points_image1_normalized": pts1_norm.tolist(),
        "points_image2_normalized": pts2_norm.tolist(),
        "essential_matrix": E.tolist(),
        "rotation_matrix": R.tolist(),
        "translation_direction": t.tolist(),
        "positive_depth_count": int(positive_depth_count),
        "sampson_error_mean": float(np.mean(epipolar_errors)),
        "sampson_error_std": float(np.std(epipolar_errors)),
        "sampson_error_max": float(np.max(epipolar_errors)),
        "scale_factor": None if scale_factor is None else float(scale_factor),
        "translation_scaled": None if t_scaled is None else t_scaled.tolist()
    }

    json_path = os.path.join(args.outdir, "eight_point_result.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=4)

    npz_path = os.path.join(args.outdir, "eight_point_result.npz")
    np.savez(
        npz_path,
        K=K,
        dist_coeffs=np.array([]) if dist is None else dist,
        pts1_px=pts1_px,
        pts2_px=pts2_px,
        pts1_norm=pts1_norm,
        pts2_norm=pts2_norm,
        E=E,
        R=R,
        t=t,
        points_3d=points_3d,
        epipolar_errors=epipolar_errors
    )

    match_vis_path = os.path.join(args.outdir, "matches.png")
    save_match_visualization(img1, img2, pts1_px, pts2_px, match_vis_path)

if __name__ == "__main__":
    main()