using System.Linq;
using TensorFlowLite;
using Unity.Barracuda;
using UnityEngine;
using UnityEngine.UI;

public class DepthSensor : MonoBehaviour
{
    [SerializeField] private NNModel _monoDepthONNX;
    [SerializeField] private RawImage _sourceImageView;
    [SerializeField] private RawImage _destinationImageView;
    private Model m_RuntimeModel;
    private IWorker worker;
    private WebCamTexture _webCamTexture;
    private RenderTexture outputRenderTexture;
    private int channelCount = 3;
    private TextureResizer.ResizeOptions _options;
    private TextureResizer _resizer;
    private RenderTexture frame;
    private Texture2D inputTexture;
    private Texture2D depthTexture;
    private Rect region;
    private int modelwidth = 224;
    private int modelheight = 224;


    private Vector3[] vertices;
    private int[] triangles;
    private Mesh mesh;
    private Color[] colors;

    private void Start()
    {
        InitWebCamFeed();
        InitBarracuda();
        InitResizerAndTextures();
        InitPointCloudMesh();
    }

    private void InitBarracuda()
    {
        m_RuntimeModel = ModelLoader.Load(_monoDepthONNX);
        worker = WorkerFactory.CreateComputeWorker(m_RuntimeModel);
    }

    private void InitWebCamFeed()
    {
        _webCamTexture = new WebCamTexture(620, 480, 30);
        _sourceImageView.texture = _webCamTexture;
        _webCamTexture.Play();
    }

    private void InitPointCloudMesh()
    {
        vertices = new Vector3[modelwidth * modelheight];
        triangles = MakeMeshTriangles();
        mesh = new Mesh();
        colors = new Color[modelwidth * modelheight];
    }

    private void InitResizerAndTextures()
    {
        _resizer = new TextureResizer();
        _options = new TextureResizer.ResizeOptions();
        _options.width = modelwidth;
        _options.height = modelheight;
        inputTexture = new Texture2D(modelwidth, modelheight, TextureFormat.RGB24, false);
        depthTexture = new Texture2D(modelwidth, modelheight, TextureFormat.RGB24, false);
        region = new Rect(0, 0, modelwidth, modelheight);

        var renderer = GetComponent<Renderer>();
        renderer.material.SetTexture("point_texture", inputTexture);
    }


    private void Update()
    {
        Color[] pixels = _webCamTexture.GetPixels();

        if (pixels.Length >= (modelwidth * modelheight))
        {
            ResizeWebCamFeedToInputTexture();

            var tensor = new Tensor(inputTexture);
            // inference
            var output = worker.Execute(tensor).PeekOutput();
            float[] depth = output.AsFloats();
            PrepareDepthTextureFromFloats(depth);
            _destinationImageView.texture = depthTexture;

            UpdatePointCloudMeshFilter();
            tensor.Dispose();
        }
    }

    private void ResizeWebCamFeedToInputTexture()
    {
        //Resize the webcam texture into the input shape dimensions
        RenderTexture tex = _resizer.Resize(_webCamTexture, _options);
        RenderTexture.active = tex;
        inputTexture.ReadPixels(region, 0, 0);
        RenderTexture.active = null;
        inputTexture.Apply();
        _sourceImageView.texture = inputTexture;
    }

    private void PrepareDepthTextureFromFloats(float[] depth)
    {
        var min = depth.Min();
        var max = depth.Max();
        foreach (var pix in depth.Select((v, i) => new { v, i }))
        {
            var x = pix.i % modelwidth;
            var y = pix.i / modelwidth;
            var invY = modelheight - y - 1;

            // normalize depth value
            var val = (pix.v - min) / (max - min);
            depthTexture.SetPixel(x, y, new Color(val, 0.59f * val, 0.11f * val));
            var worldPos = new Vector3(x / (modelwidth / 0.9f), y / (modelheight / 0.9f), val);
            vertices[y * modelwidth + x] = worldPos;
            colors[y * modelwidth + x] = inputTexture.GetPixel(x, invY);
        }
        depthTexture.Apply();
    }

    private void UpdatePointCloudMeshFilter()
    {
        mesh.SetVertices(vertices);
        mesh.SetColors(colors);
        mesh.SetTriangles(triangles, 0);
        mesh.SetIndices(mesh.GetIndices(0), MeshTopology.Points, 0);
        GetComponent<MeshFilter>().sharedMesh = mesh;
    }

    public int[] MakeMeshTriangles()
    {
        var triangles = new int[(modelwidth - 1) * (modelheight - 1) * 6];
        for (int y = 0; y < modelheight - 1; ++y)
        {
            for (int x = 0; x < modelwidth - 1; ++x)
            {
                int ul = y * modelwidth + x;
                int ur = y * modelwidth + x + 1;
                int ll = (y + 1) * modelwidth + x;
                int lr = (y + 1) * modelwidth + x + 1;

                int offset = (y * (modelwidth - 1) + x) * 6;

                triangles[offset + 0] = ll;
                triangles[offset + 1] = ul;
                triangles[offset + 2] = ur;
                triangles[offset + 3] = ll;
                triangles[offset + 4] = ur;
                triangles[offset + 5] = lr;

            }
        }

        return triangles;
    }
}
