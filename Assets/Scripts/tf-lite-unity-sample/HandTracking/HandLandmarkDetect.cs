﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;


namespace TensorFlowLite
{
    public class HandLandmarkDetect : BaseImagePredictor<float>
    {
        public class Result
        {
            public float score;
            public Vector3[] joints;
        }

        public enum Dimension
        {
            TWO,
            THREE,
        }

        public const int JOINT_COUNT = 21;

        private float[] output0 = new float[JOINT_COUNT * 2]; // keypoint
        private float[] output1 = new float[1]; // hand flag
        private Result result;
        private Matrix4x4 cropMatrix;

        public Dimension Dim { get; private set; }
        public Vector2 PalmShift { get; set; } = new Vector2(0, -0.2f);
        public float PalmScale { get; set; } = 2.8f;

        public HandLandmarkDetect(string modelPath) : base(modelPath, true)
        {
            var out0info = interpreter.GetOutputTensorInfo(0);
            switch (out0info.shape[1])
            {
                case JOINT_COUNT * 2:
                    Dim = Dimension.TWO;
                    break;
                case JOINT_COUNT * 3:
                    Dim = Dimension.THREE;
                    break;
                default:
                    throw new System.NotSupportedException();
            }
            output0 = new float[out0info.shape[1]];

            result = new Result()
            {
                score = 0,
                joints = new Vector3[JOINT_COUNT],
            };
        }

        public override void Invoke(Texture inputTex)
        {
            throw new System.NotImplementedException("Use Invoke(Texture inputTex, PalmDetect.Palm palm)");
        }

        public void Invoke(Texture inputTex, PalmDetect.Palm palm)
        {
            var options = resizeOptions;

            cropMatrix = resizer.VertexTransfrom = CalcPalmMatrix(ref palm, PalmShift, PalmScale);
            resizer.UVRect = TextureResizer.GetTextureST(inputTex, options);
            RenderTexture rt = resizer.ApplyResize(inputTex, options.width, options.height);
            ToTensor(rt, input0, false);

            //
            interpreter.SetInputTensorData(0, input0);
            interpreter.Invoke();
            interpreter.GetOutputTensorData(0, output0);
            interpreter.GetOutputTensorData(1, output1);
        }

        public Result GetResult()
        {
            // Normalize 0 ~ 255 => 0.0 ~ 1.0
            const float SCALE = 1f / 255f;
            var mtx = cropMatrix.inverse;

            result.score = output1[0];
            if (Dim == Dimension.TWO)
            {
                for (int i = 0; i < JOINT_COUNT; i++)
                {
                    result.joints[i] = mtx.MultiplyPoint3x4(new Vector3(
                        output0[i * 2],
                        output0[i * 2 + 1],
                        0
                    ) * SCALE);
                }
            }
            else
            {
                for (int i = 0; i < JOINT_COUNT; i++)
                {
                    result.joints[i] = mtx.MultiplyPoint3x4(new Vector3(
                        output0[i * 3],
                        output0[i * 3 + 1],
                        output0[i * 3 + 2]
                    ) * SCALE);
                }
            }
            return result;
        }


        private static readonly Matrix4x4 PUSH_MATRIX = Matrix4x4.Translate(new Vector3(0.5f, 0.5f, 0));
        private static readonly Matrix4x4 POP_MATRIX = Matrix4x4.Translate(new Vector3(-0.5f, -0.5f, 0));
        private static Matrix4x4 CalcPalmMatrix(ref PalmDetect.Palm palm, Vector2 shift, float scale)
        {
            // Calc rotation based on 
            // Center of wrist - Middle finger
            const float RAD_90 = 90f * Mathf.PI / 180f;
            var vec = palm.keypoints[2] - palm.keypoints[0];
            Quaternion rotation = Quaternion.Euler(0, 0, -(RAD_90 + Mathf.Atan2(vec.y, vec.x)) * Mathf.Rad2Deg);

            // Calc hand scale
            float handScale = Mathf.Max(palm.rect.width, palm.rect.height) * scale;

            // Calc hand center position
            Vector2 center = palm.rect.center + new Vector2(-0.5f, -0.5f);
            center = (Vector2)(rotation * center);
            center += (shift * handScale);
            center /= handScale;

            Matrix4x4 trs = Matrix4x4.TRS(
                               new Vector3(-center.x, -center.y, 0),
                               rotation,
                               new Vector3(1 / handScale, -1 / handScale, 1)
                            );
            return PUSH_MATRIX * trs * POP_MATRIX;
        }
    }
}