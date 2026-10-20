import math
import sys
import types
import unittest
from unittest import mock

import einops
import einops.layers.torch
import numpy as np
import torch

from data_juicer.utils.unittest_utils import DataJuicerTestCaseBase, TEST_TAG


def _lightweight_import_mocks():
    timm = types.ModuleType("timm")
    timm.layers = types.SimpleNamespace(
        drop_path=lambda x, drop_prob=0.0, training=False: (
            torch.zeros_like(x) if training and drop_prob == 1.0 else x
        ),
        to_2tuple=lambda x: x if isinstance(x, tuple) else (x, x),
        trunc_normal_=torch.nn.init.trunc_normal_,
    )

    smplx = types.ModuleType("smplx")
    smplx.MANOLayer = torch.nn.Module
    smplx.utils = types.SimpleNamespace(MANOOutput=object, to_tensor=torch.as_tensor)
    smplx.vertex_ids = types.SimpleNamespace(vertex_ids={"mano": {}})
    smplx.lbs = types.SimpleNamespace(vertices2joints=lambda regressor, vertices: vertices)

    lightning = types.ModuleType("pytorch_lightning")
    lightning.LightningModule = torch.nn.Module
    rank_zero_only = lambda fn: fn
    lightning.utilities = types.SimpleNamespace(
        rank_zero_only=rank_zero_only,
        rank_zero=types.SimpleNamespace(rank_zero_only=rank_zero_only),
    )
    return {
        "timm": timm,
        "smplx": smplx,
        "pytorch_lightning": lightning,
    }


with mock.patch.dict(sys.modules, _lightweight_import_mocks()):
    from data_juicer.ops.common import dwpose_func as dw
    from data_juicer.ops.common import hawor_func as hw
    from data_juicer.ops.common import hawor_func_vit as hv


class HaworGeometryTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_transform_crop_boxes_and_helpers(self):
        center = np.array([4.0, 4.0])
        point = np.array([3.0, 5.0])
        transformed = hw.transform(point, center, 0.04, (8, 8), rot=25, asint=False)
        restored = hw.transform(transformed, center, 0.04, (8, 8), invert=1, rot=25, asint=False)
        np.testing.assert_allclose(restored, point, atol=1e-6)
        self.assertEqual(hw.transform(point, center, 0.04, (8, 8)).dtype.kind, "i")

        image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
        cropped = hw.crop(image, center, 0.04, (6, 4))
        self.assertEqual(cropped.shape, (6, 4, 3))
        rotated = hw.crop(image, center, 0.04, (6, 6), rot=15)
        self.assertEqual(rotated.shape, (6, 6, 3))

        centers, scales = hw.boxes_2_cs(np.array([[0, 2, 10, 6], [2, 0, 4, 8]], dtype=float))
        np.testing.assert_allclose(centers, [[5, 4], [3, 4]])
        np.testing.assert_allclose(scales, [0.05, 0.04])
        self.assertTrue(hw.exists(0))
        self.assertEqual(hw.default(None, lambda: 7), 7)
        self.assertEqual(hw.default(3, 7), 3)

        keypoints = np.array([[1, 2, 0.9], [5, 8, 0.8], [99, 99, 0.0]])
        self.assertEqual(hw.get_keypoints_rectangle(keypoints, 0.5), (4, 6, 24))
        self.assertEqual(hw.get_keypoints_rectangle(keypoints, 1.0), (0, 0, 0))

    @TEST_TAG("standalone")
    def test_lightweight_keypoint_renderers(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        hand = np.c_[
            np.linspace(8, 55, 21),
            np.linspace(10, 50, 21),
            np.ones(21),
        ]
        rendered_hand = hw.render_openpose(image, hand)
        self.assertEqual(rendered_hand.shape, image.shape)
        self.assertGreater(int(rendered_hand.sum()), 0)
        rendered_landmarks = hw.render_openpose_landmarks(image, hand)
        self.assertGreater(int(rendered_landmarks.sum()), 0)
        rendered_confidence = hw.render_hand_keypoints(
            image, hand, use_confidence=True, map_fn=lambda confidence: confidence
        )
        self.assertGreater(int(rendered_confidence.sum()), 0)

        body = np.c_[
            np.linspace(5, 58, 25),
            np.linspace(8, 52, 25),
            np.ones(25),
        ]
        rendered_body = hw.render_body_keypoints(image, body)
        self.assertEqual(rendered_body.shape, image.shape)
        self.assertGreater(int(rendered_body.sum()), 0)

    @TEST_TAG("standalone")
    def test_projection_rotation_and_quaternion_conversions(self):
        points = torch.tensor([[[1.0, 2.0, 2.0], [0.0, 0.0, 1.0]]])
        projected = hw.perspective_projection(
            points,
            torch.eye(3).unsqueeze(0),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([2.0]),
            torch.tensor([[10.0, 20.0]]),
        )
        torch.testing.assert_close(projected[0, 0], torch.tensor([10.666666, 21.333334]), atol=1e-5, rtol=0)
        distorted = hw.perspective_projection(
            points, None, None, torch.tensor([2.0]), torch.zeros(1, 2), torch.zeros(1, 5)
        )
        torch.testing.assert_close(distorted[0, 0], torch.tensor([1.0, 2.0]))

        identity = torch.eye(3).unsqueeze(0)
        torch.testing.assert_close(hw.avg_rot(identity.repeat(2, 1, 1)), identity[0])
        torch.testing.assert_close(hw.rot9d_to_rotmat(identity.reshape(1, 9)), identity)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        torch.testing.assert_close(hw.quat_to_rotmat(quat), identity)
        torch.testing.assert_close(hw.quaternion_to_rotation_matrix(quat), identity)
        torch.testing.assert_close(hw.rotation_matrix_to_quaternion(identity), quat)
        torch.testing.assert_close(hw.rotation_matrix_to_angle_axis(identity), torch.zeros(1, 3))
        torch.testing.assert_close(hw.quaternion_to_angle_axis(quat), torch.zeros(1, 3))
        with self.assertRaises(TypeError):
            hw.quaternion_to_angle_axis([1, 0, 0, 0])
        with self.assertRaises(ValueError):
            hw.quaternion_to_angle_axis(torch.ones(3))

        self.assertEqual(hw.rotmat_to_rot6d(identity).shape, (1, 6))
        canonical_rot6 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        torch.testing.assert_close(hw.rot6d_to_rotmat(canonical_rot6), identity)
        angle_axis = torch.tensor([[0.0, 0.0, math.pi / 2], [0.0, 0.0, 0.0]])
        quats = hw.angle_axis_to_quaternion(angle_axis)
        mats = hw.angle_axis_to_rotation_matrix(angle_axis)
        torch.testing.assert_close(mats, hw.aa_to_rotmat(angle_axis), atol=1e-6, rtol=1e-6)
        product = hw.quaternion_mul(quats, hw.quaternion_inverse(quats))
        torch.testing.assert_close(product, quat.repeat(2, 1), atol=1e-6, rtol=1e-6)
        halfway = hw.quaternion_slerp(torch.tensor([0.5]), quat, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
        self.assertAlmostEqual(float(halfway.norm()), 1.0, places=6)
        negative_path = hw.quaternion_slerp(
            torch.tensor([0.25]), quat, torch.tensor([[-math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)]])
        )
        self.assertAlmostEqual(float(negative_path.norm()), 1.0, places=6)
        self.assertGreater(float(negative_path[0, 0]), 0.0)

    @TEST_TAG("standalone")
    def test_translation_parsing_and_interpolation(self):
        rng = np.random.default_rng(4)
        points = rng.normal(size=(24, 3))
        points[:, 2] += 5
        translation = np.array([0.2, -0.1, 2.0])
        focal, size = 500.0, 224.0
        camera = points + translation
        joints = camera[:, :2] / camera[:, 2:] * focal + size / 2
        estimated = hw.estimate_translation_np(points, joints, np.ones(24), focal, size)
        np.testing.assert_allclose(estimated, translation, atol=1e-5)
        batched = hw.estimate_translation(
            torch.tensor(points[None], dtype=torch.float32),
            torch.tensor(np.c_[joints, np.ones(24)][None], dtype=torch.float32),
            focal,
            size,
        )
        np.testing.assert_allclose(batched.numpy()[0], translation, atol=2e-4)

        frames = np.array([0, 1, 2, 5, 6, 9, 10, 11])
        boxes = np.arange(40).reshape(8, 5)
        frame_chunks, box_chunks = hw.parse_chunks(frames, boxes, min_len=2)
        self.assertEqual([x.tolist() for x in frame_chunks], [[0, 1, 2], [5, 6], [9, 10, 11]])
        self.assertEqual([len(x) for x in box_chunks], [3, 2, 3])

        missing = np.array([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [3, 4, 5, 6, 7]], dtype=float)
        filled = hw.interpolate_bboxes(missing)
        np.testing.assert_allclose(filled[0], [0, 1, 2, 3, 4])
        np.testing.assert_allclose(filled[2], [2, 3, 4, 5, 6])
        self.assertIs(hw.interpolate_bboxes(filled), filled)


class HaworNetworksTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        self.einops_patch = mock.patch.object(hw, "einops", einops)
        self.einops_patch.start()

    def tearDown(self):
        self.einops_patch.stop()
        super().tearDown()

    @TEST_TAG("standalone")
    def test_normalization_frequency_attention_and_transformers(self):
        torch.manual_seed(0)
        self.assertIsInstance(hw.normalization_layer("batch", 4), torch.nn.BatchNorm1d)
        self.assertIsInstance(hw.normalization_layer("layer", 4), torch.nn.LayerNorm)
        self.assertIsInstance(hw.normalization_layer(None, 4), torch.nn.Identity)
        with self.assertRaises(ValueError):
            hw.normalization_layer("bad", 4)
        with self.assertRaises(AssertionError):
            hw.normalization_layer("ada", 4)
        with self.assertRaises(ValueError):
            hw.AdaptiveLayerNorm1D(0, 2)

        ada = hw.AdaptiveLayerNorm1D(4, 2)
        x = torch.randn(2, 3, 4)
        conditioned = ada(x, torch.ones(2, 2))
        torch.testing.assert_close(conditioned, torch.nn.functional.layer_norm(x, (4,)))
        embedded = hw.FrequencyEmbedder(2, 1)(torch.tensor([0.0, 1.0]))
        self.assertEqual(embedded.shape, (2, 5))
        self.assertEqual(hw.FrequencyEmbedder(2, 1)(torch.ones(2, 3)).shape, (2, 15))

        self.assertEqual(hw.FeedForward(4, 8)(x).shape, x.shape)
        self.assertEqual(hw.Attention(4, heads=1, dim_head=4)(x).shape, x.shape)
        self.assertEqual(hw.Attention(4, heads=2, dim_head=3)(x).shape, x.shape)
        self.assertEqual(hw.CrossAttention(4, context_dim=6, heads=2, dim_head=2)(x, torch.randn(2, 2, 6)).shape, x.shape)
        self.assertEqual(hw.Transformer(4, 2, 2, 2, 8)(x).shape, x.shape)
        cross = hw.TransformerCrossAttn(4, 2, 2, 2, 8, context_dim=6)
        self.assertEqual(cross(x, context=torch.randn(2, 2, 6)).shape, x.shape)
        with self.assertRaises(ValueError):
            cross(x, context_list=[torch.randn(2, 2, 6)])

        adaptive_transformer = hw.Transformer(4, 1, 1, 4, 8, norm="ada", norm_cond_dim=2)
        self.assertEqual(adaptive_transformer(x, torch.ones(2, 2)).shape, x.shape)

    @TEST_TAG("standalone")
    def test_dropout_encoders_and_positional_encoding(self):
        x = torch.ones(2, 4, 3)
        with self.assertRaises(ValueError):
            hw.DropTokenDropout(-0.1)
        drop = hw.DropTokenDropout(1.0)
        drop.train()
        self.assertEqual(drop(x).shape, (2, 0, 3))
        drop.eval()
        self.assertEqual(drop(x).shape, x.shape)
        zero = hw.ZeroTokenDropout(1.0)
        zero.train()
        self.assertEqual(int(torch.count_nonzero(zero(x.clone()))), 0)

        for location in ("input", "token", "token_afterpos"):
            encoder = hw.TransformerEncoder(4, 3, 4, 1, 1, 8, dim_head=4, emb_dropout_loc=location)
            self.assertEqual(encoder(torch.randn(2, 4, 3)).shape, (2, 4, 4))
        freq_encoder = hw.TransformerEncoder(4, 2, 4, 1, 1, 8, dim_head=4, token_pe_numfreq=2)
        self.assertEqual(freq_encoder(torch.randn(2, 4, 2)).shape, (2, 4, 4))
        with self.assertRaises(ValueError):
            hw.TransformerEncoder(4, 3, 4, 1, 1, 8, emb_dropout_type="bad")

        decoder = hw.TransformerDecoder(3, 4, 4, 1, 1, 8, dim_head=4, context_dim=5, emb_dropout_type="normal")
        self.assertEqual(decoder(torch.randn(2, 3, 4), context=torch.randn(2, 2, 5)).shape, (2, 3, 4))
        self.assertIsInstance(hw.TransformerDecoder(3, 4, 4, 1, 1, 8, skip_token_embedding=True).to_token_embedding, torch.nn.Identity)
        with self.assertRaises(ValueError):
            hw.TransformerDecoder(3, 3, 4, 1, 1, 8, skip_token_embedding=True)

        pe = hw.PositionalEncoding(4, dropout=0.0, max_len=5)
        encoded = pe(torch.zeros(3, 2, 4))
        torch.testing.assert_close(encoded[0, :, 0::2], torch.zeros(2, 2))
        temporal = hw.temporal_attention(in_dim=4, out_dim=4, hdim=4, nlayer=1, nhead=1, residual=True).eval()
        self.assertEqual(temporal(torch.randn(2, 3, 4)).shape, (2, 3, 4))

    @TEST_TAG("standalone")
    def test_hawor_camera_helpers_without_model(self):
        model = hw.HAWOR.__new__(hw.HAWOR)
        torch.nn.Module.__init__(model)
        model.crop_size = 200
        center = torch.tensor([[100.0, 80.0]])
        scale = torch.tensor([1.0])
        focal = torch.tensor([500.0])
        image_center = torch.tensor([[90.0, 70.0]])
        camera = torch.tensor([[2.0, 0.1, -0.2]])
        trans = model.get_trans(camera, center, scale, focal, image_center)
        torch.testing.assert_close(trans, torch.tensor([[[0.15, -0.15, 2.5]]]))
        bbox = model.bbox_est(center, scale, focal, image_center)
        torch.testing.assert_close(bbox, torch.tensor([[0.056, 0.056, 2.6666667]]), atol=1e-6, rtol=1e-6)
        points = torch.tensor([[[0.0, 0.0, 0.0]]])
        full, crop = model.project(points, camera, center, scale, focal, image_center, return_full=True)
        self.assertEqual(full.shape, (1, 1, 2))
        self.assertEqual(crop.shape, (1, 1, 2))


class HaworVitTest(DataJuicerTestCaseBase):

    def setUp(self):
        super().setUp()
        layers = types.SimpleNamespace(
            drop_path=lambda x, drop_prob=0.0, training=False: (
                torch.zeros_like(x) if training and drop_prob == 1.0 else x
            ),
            to_2tuple=lambda x: x if isinstance(x, tuple) else (x, x),
            trunc_normal_=torch.nn.init.trunc_normal_,
        )
        self.timm_patch = mock.patch.object(hv, "timm", types.SimpleNamespace(layers=layers))
        self.timm_patch.start()

    def tearDown(self):
        self.timm_patch.stop()
        super().tearDown()

    @TEST_TAG("standalone")
    def test_abs_pos_layers_patch_and_hybrid_embed(self):
        pos = torch.arange(5 * 3, dtype=torch.float32).reshape(1, 5, 3)
        self.assertEqual(hv.get_abs_pos(pos, 2, 2, 2, 2).shape, (1, 5, 3))
        resized = hv.get_abs_pos(pos, 1, 2, 2, 2)
        self.assertEqual(resized.shape, (1, 3, 3))
        no_cls = hv.get_abs_pos(pos[:, 1:], 1, 2, 2, 2, has_cls_token=False)
        self.assertEqual(no_cls.shape, (1, 2, 3))

        drop = hv.DropPath(1.0)
        drop.train()
        self.assertEqual(int(torch.count_nonzero(drop(torch.ones(2, 3)))), 0)
        self.assertEqual(drop.extra_repr(), "p=1.0")
        self.assertEqual(hv.Mlp(4, 6, 5)(torch.randn(2, 3, 4)).shape, (2, 3, 5))
        self.assertEqual(hv.Attention(6, num_heads=2, attn_head_dim=2)(torch.randn(2, 3, 6)).shape, (2, 3, 6))
        self.assertEqual(hv.Block(6, 2, mlp_ratio=2)(torch.randn(2, 3, 6)).shape, (2, 3, 6))

        patch = hv.PatchEmbed(img_size=(8, 8), patch_size=4, embed_dim=6)
        tokens, shape = patch(torch.randn(2, 3, 8, 8))
        self.assertEqual((tokens.shape, shape), (torch.Size([2, 9, 6]), (3, 3)))

        class ListBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3, stride=2, padding=1)

            def forward(self, value):
                return [self.conv(value)]

        hybrid = hv.HybridEmbed(ListBackbone(), img_size=8, embed_dim=4)
        self.assertEqual(hybrid(torch.randn(2, 3, 8, 8)).shape, (2, 16, 4))

    @TEST_TAG("standalone")
    def test_small_vit_forward_freezing_and_checkpoint(self):
        torch.manual_seed(2)
        model = hv.ViT(
            img_size=(8, 8), patch_size=8, embed_dim=8, depth=2, num_heads=2,
            mlp_ratio=2, qkv_bias=True, drop_path_rate=0.1, frozen_stages=0,
            freeze_attn=True, freeze_ffn=True,
        )
        model.init_weights()
        self.assertEqual(model.get_num_layers(), 2)
        self.assertEqual(model.no_weight_decay(), {"pos_embed", "cls_token"})
        output = model(torch.randn(1, 3, 8, 8))
        self.assertEqual(output.shape, (1, 8, 1, 1))
        self.assertTrue(all(not p.requires_grad for p in model.patch_embed.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.blocks[0].attn.parameters()))
        model.train()
        self.assertFalse(model.patch_embed.training)

        checkpointed = hv.ViT(img_size=8, patch_size=8, embed_dim=8, depth=1, num_heads=2, use_checkpoint=True)
        with mock.patch.object(torch.utils.checkpoint, "checkpoint", side_effect=lambda fn, x: fn(x)) as checkpoint:
            self.assertEqual(checkpointed(torch.randn(1, 3, 8, 8)).shape, (1, 8, 1, 1))
            checkpoint.assert_called_once()


class DWPoseTest(DataJuicerTestCaseBase):

    @TEST_TAG("standalone")
    def test_nms_postprocess_and_detector_inference(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], dtype=float)
        self.assertEqual(dw.nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5), [0, 2])
        scores = np.array([[0.9, 0.0], [0.8, 0.0], [0.0, 0.7]])
        dets = dw.multiclass_nms(boxes, scores, 0.5, 0.1)
        self.assertEqual(dets.shape, (2, 6))
        self.assertIsNone(dw.multiclass_nms(boxes, np.zeros((3, 2)), 0.5, 0.1))

        count = sum((8 // s) * (8 // s) for s in (8, 16, 32))
        outputs = np.zeros((1, count, 4))
        processed = dw.demo_postprocess(outputs, (8, 8))
        np.testing.assert_allclose(processed[0, 0], [0, 0, 8, 8])
        p6_count = sum((64 // s) * (64 // s) for s in (8, 16, 32, 64))
        self.assertEqual(dw.demo_postprocess(np.zeros((1, p6_count, 4)), (64, 64), p6=True).shape[1], p6_count)

        predictions = np.zeros((1, count, 7), dtype=float)
        predictions[0, 0] = [4, 4, 1, 1, 0.9, 0.9, 0.0]
        session = mock.Mock()
        session.get_inputs.return_value = [types.SimpleNamespace(name="images")]
        session.run.return_value = [predictions]
        with mock.patch.object(dw, "preprocess_det", return_value=(np.zeros((3, 8, 8)), 1.0)), mock.patch.object(
            dw, "demo_postprocess", side_effect=lambda value, size: value
        ):
            detected = dw.inference_detector(session, np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(detected.shape, (1, 4))
        predictions[..., 4:] = 0
        with mock.patch.object(dw, "preprocess_det", return_value=(np.zeros((3, 8, 8)), 1.0)), mock.patch.object(
            dw, "demo_postprocess", side_effect=lambda value, size: value
        ):
            self.assertEqual(dw.inference_detector(session, np.zeros((8, 8, 3), dtype=np.uint8)).size, 0)

    @TEST_TAG("standalone")
    def test_bbox_affine_preprocess_and_simcc(self):
        single_center, single_scale = dw.bbox_xyxy2cs(np.array([0, 0, 8, 4]), padding=1.25)
        np.testing.assert_allclose(single_center, [4, 2])
        np.testing.assert_allclose(single_scale, [10, 5])
        centers, scales = dw.bbox_xyxy2cs(np.array([[0, 0, 4, 8], [2, 2, 6, 6]]))
        self.assertEqual(centers.shape, (2, 2))
        np.testing.assert_allclose(dw._fix_aspect_ratio(np.array([10.0, 5.0]), 1.0), [10, 10])
        np.testing.assert_allclose(dw._fix_aspect_ratio(np.array([4.0, 8.0]), 0.5), [4, 8])
        np.testing.assert_allclose(dw._rotate_point(np.array([1.0, 0.0]), math.pi / 2), [0, 1], atol=1e-7)
        np.testing.assert_allclose(dw._get_3rd_point(np.array([1, 0]), np.array([0, 0])), [0, 1])

        warp = dw.get_warp_matrix(np.array([4.0, 4.0]), np.array([8.0, 8.0]), 0, (8, 8))
        inverse = dw.get_warp_matrix(np.array([4.0, 4.0]), np.array([8.0, 8.0]), 0, (8, 8), inv=True)
        np.testing.assert_allclose(warp, inverse, atol=1e-6)
        image = np.arange(8 * 4 * 3, dtype=np.uint8).reshape(8, 4, 3)
        affine, fixed_scale = dw.top_down_affine((4, 8), np.array([4.0, 8.0]), np.array([2.0, 4.0]), image)
        self.assertEqual(affine.shape, (8, 4, 3))
        np.testing.assert_allclose(fixed_scale, [4, 8])

        preprocessed, ratio = dw.preprocess_det(image, (16, 16))
        self.assertEqual(preprocessed.shape, (3, 16, 16))
        self.assertEqual(ratio, 2.0)
        gray, _ = dw.preprocess_det(np.zeros((4, 8), dtype=np.uint8), (8, 8), swap=(0, 1))
        self.assertEqual(gray.shape, (8, 8))
        pose_images, pose_centers, pose_scales = dw.preprocess_pose(image, [], input_size=(4, 8))
        self.assertEqual((len(pose_images), len(pose_centers), len(pose_scales)), (1, 1, 1))

        simcc_x = np.array([[[0, 1, 0, 0], [0, -1, -2, -3]]], dtype=float)
        simcc_y = np.array([[[0, 0, 2, 0], [0, -1, -2, -3]]], dtype=float)
        locations, values = dw.get_simcc_maximum(simcc_x, simcc_y)
        np.testing.assert_allclose(locations[0, 0], [1, 2])
        np.testing.assert_allclose(locations[0, 1], [-1, -1])
        decoded, decoded_values = dw.decode(simcc_x, simcc_y, 2)
        np.testing.assert_allclose(decoded[0, 0], [0.5, 1])
        np.testing.assert_allclose(decoded_values, values)
        keys, vals = dw.postprocess([[simcc_x, simcc_y]], (4, 4), [np.array([4, 4])], [np.array([8, 8])], 2)
        self.assertEqual((keys.shape, vals.shape), ((1, 2, 2), (1, 2)))

    @TEST_TAG("standalone")
    def test_session_resize_padding_transfer_and_detection_helpers(self):
        session = mock.Mock()
        session.get_inputs.return_value = [types.SimpleNamespace(name="input", shape=[1, 3, 8, 4])]
        session.get_outputs.return_value = [types.SimpleNamespace(name="x"), types.SimpleNamespace(name="y")]
        session.run.return_value = [np.ones((1, 1, 8)), np.ones((1, 1, 16))]
        outputs = dw.inference(session, [np.zeros((8, 4, 3))])
        self.assertEqual(len(outputs), 1)
        with mock.patch.object(dw, "preprocess_pose", return_value=([np.zeros((8, 4, 3))], [np.array([2, 4])], [np.array([4, 8])])):
            keypoints, scores = dw.inference_pose(session, [], np.zeros((8, 4, 3)))
        self.assertEqual((keypoints.shape, scores.shape), ((1, 1, 2), (1, 1)))

        rgb = np.zeros((4, 6, 3), dtype=np.uint8)
        self.assertEqual(dw.smart_resize(rgb, (2, 3)).shape, (2, 3, 3))
        self.assertEqual(dw.smart_resize(rgb, (8, 12)).shape, (8, 12, 3))
        multispectral = np.zeros((4, 6, 4), dtype=np.uint8)
        self.assertEqual(dw.smart_resize(multispectral, (2, 3)).shape, (2, 3, 4))
        self.assertEqual(dw.smart_resize_k(multispectral, 2, 2).shape, (8, 12, 4))
        padded, pad = dw.padRightDownCorner(np.ones((3, 5, 3)), 4, 7)
        self.assertEqual((padded.shape, pad), ((4, 8, 3), [0, 0, 1, 3]))
        self.assertTrue(np.all(padded[-1] == 7))

        model = torch.nn.Sequential(torch.nn.Linear(2, 2))
        source = {"weight": torch.ones(2, 2), "bias": torch.zeros(2)}
        transferred = dw.transfer(model, source)
        self.assertEqual(set(transferred), {"0.weight", "0.bias"})
        self.assertEqual(dw.npmax(np.array([[1, 2], [5, 3]])), (1, 0))

        candidate = np.zeros((18, 3), dtype=float)
        candidate[2, :2], candidate[3, :2], candidate[4, :2] = [20, 20], [40, 40], [70, 70]
        subset = np.full((1, 18), -1)
        subset[0, [2, 3, 4]] = [2, 3, 4]
        hands = dw.handDetect(candidate, subset, np.zeros((100, 100, 3)))
        self.assertEqual(len(hands), 1)
        self.assertFalse(hands[0][3])

        candidate[0, :2], candidate[14, :2] = [50, 50], [60, 50]
        subset[0, 0], subset[0, 14] = 0, 14
        faces = dw.faceDetect(candidate, subset, np.zeros((100, 100, 3)))
        self.assertEqual(len(faces), 1)

    @TEST_TAG("standalone")
    def test_drawing_helpers(self):
        candidate = np.stack([np.linspace(0.1, 0.9, 18), np.linspace(0.9, 0.1, 18)], axis=1)
        subset = np.arange(18, dtype=float)[None]
        canvas = np.zeros((64, 64, 3), dtype=np.uint8)
        body = dw.draw_bodypose(canvas.copy(), candidate, subset)
        self.assertGreater(int(body.sum()), 0)

        hand = np.stack([np.linspace(0.1, 0.9, 21), np.linspace(0.2, 0.8, 21)], axis=1)
        hand_canvas = dw.draw_handpose(canvas.copy(), [hand])
        face_canvas = dw.draw_facepose(canvas.copy(), [hand[:5]])
        self.assertGreater(int(hand_canvas.sum()), 0)
        self.assertGreater(int(face_canvas.sum()), 0)
        pose = {
            "bodies": {"candidate": candidate, "subset": subset},
            "hands": [hand],
            "faces": [hand[:5]],
        }
        self.assertEqual(dw.draw_pose(pose, 64, 64).shape, canvas.shape)

    @TEST_TAG("standalone")
    def test_wholebody_and_detector_wrappers_with_mocked_models(self):
        fake_session = mock.Mock()
        runtime = types.SimpleNamespace(InferenceSession=mock.Mock(return_value=fake_session))
        with mock.patch.object(dw, "onnxruntime", runtime):
            wholebody = dw.Wholebody("det.onnx", "pose.onnx", "cpu")
        self.assertEqual(runtime.InferenceSession.call_count, 2)

        raw_keys = np.zeros((1, 133, 2), dtype=float)
        raw_scores = np.ones((1, 133), dtype=float)
        with mock.patch.object(dw, "inference_detector", return_value=np.array([[0, 0, 1, 1]])), mock.patch.object(
            dw, "inference_pose", return_value=(raw_keys, raw_scores)
        ):
            keys, scores, boxes = wholebody(np.zeros((8, 8, 3)))
        self.assertEqual((keys.shape, scores.shape, boxes.shape), ((1, 134, 2), (1, 134), (1, 4)))

        detector = dw.DWposeDetector.__new__(dw.DWposeDetector)
        candidate = np.full((1, 134, 2), 4.0)
        confidence = np.ones((1, 134))
        detector.pose_estimation = mock.Mock(return_value=(candidate, confidence, np.array([[1, 2, 3, 4]])))
        with mock.patch.object(dw, "draw_pose", return_value=np.zeros((8, 8, 3), dtype=np.uint8)):
            body, foot, faces, hands, boxes, canvas = detector(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertEqual(body.shape, (1, 18, 2))
        self.assertEqual(foot.shape, (1, 6, 2))
        self.assertEqual(faces.shape, (1, 68, 2))
        self.assertEqual(hands.shape, (2, 21, 2))
        self.assertEqual(canvas.shape, (8, 8, 3))


if __name__ == "__main__":
    unittest.main()
