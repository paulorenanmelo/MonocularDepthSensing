﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TensorFlowLite
{
    public abstract class BaseImagePredictor<T> : System.IDisposable
    {
        protected Interpreter interpreter;
        protected int width;
        protected int height;
        protected int channels;
        protected T[,,] input0;
        protected TextureToTensor tex2tensor;
        protected TextureResizer resizer;
        protected TextureResizer.ResizeOptions resizeOptions;

        public Texture2D inputTex => tex2tensor.texture;
        public Material transformMat => resizer.material;

        public TextureResizer.ResizeOptions ResizeOptions
        {
            get => resizeOptions;
            set => resizeOptions = value;
        }

        public BaseImagePredictor(string modelPath, bool useGPU = true)
        {
            var options = new InterpreterOptions()
            {
                threads = 2//,
                //gpuDelegate = useGPU ? CreateGpuDelegate() : null,
            };
            if (useGPU) options.AddGpuDelegate(); // added
            interpreter = new Interpreter(FileUtil.LoadFile(modelPath), options);
            interpreter.LogIOInfo();
            InitInputs();

            tex2tensor = new TextureToTensor();
            resizer = new TextureResizer();
            resizeOptions = new TextureResizer.ResizeOptions()
            {
                aspectMode = TextureResizer.AspectMode.Fill,
                rotationDegree = 0,
                flipX = false,
                flipY = true,
                width = width,
                height = height,
            };
        }

        public virtual void Dispose()
        {
            interpreter?.Dispose();
            tex2tensor?.Dispose();
            resizer?.Dispose();
        }

        public abstract void Invoke(Texture inputTex);

        protected void ToTensor(Texture inputTex, float[,,] inputs)
        {
            RenderTexture tex = resizer.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs);
        }

        protected void ToTensor(RenderTexture inputTex, float[,,] inputs, bool resize)
        {
            RenderTexture tex = resize ? resizer.Resize(inputTex, resizeOptions) : inputTex;
            tex2tensor.ToTensor(tex, inputs);
        }

        protected void ToTensor(Texture inputTex, float[,,] inputs, float offset, float scale)
        {
            RenderTexture tex = resizer.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs, offset, scale);
        }

        protected void ToTensor(Texture inputTex, sbyte[,,] inputs)
        {
            RenderTexture tex = resizer.Resize(inputTex, resizeOptions);
            tex2tensor.ToTensor(tex, inputs);
        }

        private void InitInputs()
        {
            var idim0 = interpreter.GetInputTensorInfo(0).shape;
            height = idim0[1];
            width = idim0[2];
            channels = idim0[3];
            input0 = new T[height, width, channels];

            int inputCount = interpreter.GetInputTensorCount();
            for (int i = 0; i < inputCount; i++)
            {
                int[] dim = interpreter.GetInputTensorInfo(i).shape;
                interpreter.ResizeInputTensor(i, dim);
            }
            interpreter.AllocateTensors();
        }

#pragma warning disable CS0162 // Unreachable code detected 
        static IGpuDelegate CreateGpuDelegate()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            return new GlDelegate();
#elif UNITY_IOS || UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
            return new MetalDelegate(new MetalDelegate.Options()
            {
                allowPrecisionLoss = false,
                waitType = MetalDelegate.WaitType.Passive,
            });
#endif
            UnityEngine.Debug.LogWarning("GPU Delegate is not supported on this platform");
            return null;
        }
    }
#pragma warning restore CS0162 // Unreachable code detected 

}
