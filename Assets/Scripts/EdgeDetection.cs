using UnityEngine;
using UnityEngine.Video;

public class EdgeDetection : MonoBehaviour {
  [SerializeField] private ComputeShader shader;
  [SerializeField] private VideoClip videoClip;
  [SerializeField] private Vector2 quadSize = new Vector2(16, 9);

  [Header("Edge detection parameters")]
  [Range(0, 1)]
  [SerializeField] private float threshold = 0.5f;
  [SerializeField] private bool hardThreshold = true;
  [SerializeField] private Color lineColour;
  private int colourIdx = 0;

  // We need 7 materials and textures to hold the origin + modifications.
  private const int numTextures = 7;
  private Material[] materials;
  private RenderTexture[] processingTexture;
  private RenderTexture[] normalisedSobels;
  private GameObject[] quads;
  private VideoPlayer videoPlayer;

  private int kernelGreyscale, kernelThreshold, kernelThresholdFuzzy, kernelOutline;
  private int kernelHorizSobel, kernelVertSobel, kernelCombSobel;
  private int kernelHorizNorm, kernelVertNorm;
  private int threadGroupsX, threadGroupsY;

  void Awake() {
    // Get our kernel id so we don't have to look it up each time.
    kernelGreyscale = shader.FindKernel("Greyscale");
    kernelHorizSobel = shader.FindKernel("HorizSobel");
    kernelVertSobel = shader.FindKernel("VertSobel");
    kernelCombSobel = shader.FindKernel("CombSobel");
    kernelThreshold = shader.FindKernel("Threshold");
    kernelThresholdFuzzy = shader.FindKernel("ThresholdFuzzy");
    kernelOutline = shader.FindKernel("Outline");
    kernelHorizNorm = shader.FindKernel("NormaliseHorizSobel");
    kernelVertNorm = shader.FindKernel("NormaliseVertSobel");
    // We can use a depth buffer of size 0 as there is no depth.
    int depth = 0;
    processingTexture = new RenderTexture[numTextures];
    materials = new Material[numTextures];
    quads = new GameObject[numTextures];
    for (int i = 0; i < numTextures; i++) {
      // We want to deal with our textures as floats in the compute shader.
      // Linear as we want to keep the values for future procssing.
      processingTexture[i] = new RenderTexture((int)videoClip.width, (int)videoClip.height, depth,
                                               RenderTextureFormat.ARGBFloat,
                                               RenderTextureReadWrite.Linear);
      // They must have random write to be used in a compute shader.
      processingTexture[i].enableRandomWrite = true;
      processingTexture[i].Create();
      // Create the materials and assign the render textures.
      materials[i] = new Material(Shader.Find("Unlit/Texture"));
      materials[i].mainTexture = processingTexture[i];
      // Create the quads and assign the materials.
      quads[i] = GameObject.CreatePrimitive(PrimitiveType.Quad);
      quads[i].GetComponent<MeshRenderer>().material = materials[i];
      quads[i].transform.parent = gameObject.transform;
      quads[i].transform.localScale = quadSize;
      quads[i].transform.localPosition += i * (quadSize.y + 1) * Vector3.down;
    }
    // Setup our VideoPlayer to loop our clip.
    videoPlayer = gameObject.AddComponent<VideoPlayer>();
    videoPlayer.isLooping = true;
    videoPlayer.clip = videoClip;
    videoPlayer.renderMode = VideoRenderMode.RenderTexture;
    videoPlayer.targetTexture = processingTexture[0];

    // Create special textures for the sobels so they display correctly.
    normalisedSobels = new RenderTexture[2];
    for (int i = 0; i < 2; i++) {
      normalisedSobels[i] = new RenderTexture((int)videoClip.width, (int)videoClip.height, depth,
                                              RenderTextureFormat.ARGBFloat,
                                              RenderTextureReadWrite.Linear);
      // They must have random write to be used in a compute shader.
      normalisedSobels[i].enableRandomWrite = true;
      normalisedSobels[i].Create();
      //materials[2 + i].mainTexture = normalisedSobels[i];
    }
  }

  void Start() {
    // Setup the texture links each kernel needs
    shader.SetTexture(kernelGreyscale, "inputTexture", processingTexture[0]);
    shader.SetTexture(kernelGreyscale, "greyscaleTexture", processingTexture[1]);

    shader.SetTexture(kernelHorizSobel, "greyscaleTexture", processingTexture[1]);
    shader.SetTexture(kernelHorizSobel, "horizSobelTexture", processingTexture[2]);

    shader.SetTexture(kernelVertSobel, "greyscaleTexture", processingTexture[1]);
    shader.SetTexture(kernelVertSobel, "vertSobelTexture", processingTexture[3]);

    shader.SetTexture(kernelCombSobel, "horizSobelTexture", processingTexture[2]);
    shader.SetTexture(kernelCombSobel, "vertSobelTexture", processingTexture[3]);
    shader.SetTexture(kernelCombSobel, "combSobelTexture", processingTexture[4]);

    shader.SetTexture(kernelThreshold, "combSobelTexture", processingTexture[4]);
    shader.SetTexture(kernelThreshold, "thresholdTexture", processingTexture[5]);
    // Same as above as it's an alternative.
    shader.SetTexture(kernelThresholdFuzzy, "combSobelTexture", processingTexture[4]);
    shader.SetTexture(kernelThresholdFuzzy, "thresholdTexture", processingTexture[5]);

    shader.SetTexture(kernelOutline, "inputTexture", processingTexture[0]);
    shader.SetTexture(kernelOutline, "thresholdTexture", processingTexture[5]);
    shader.SetTexture(kernelOutline, "outlineTexture", processingTexture[6]);

    // The extra two functions to correctly display the Sobel filters.
    shader.SetTexture(kernelHorizNorm, "horizSobelTexture", processingTexture[2]);
    shader.SetTexture(kernelHorizNorm, "horizSobelNormalised", normalisedSobels[0]);
    shader.SetTexture(kernelVertNorm, "vertSobelTexture", processingTexture[3]);
    shader.SetTexture(kernelVertNorm, "vertSobelNormalised", normalisedSobels[1]);

    // All the compute shaders will do 8x8 pixels at a time, so we need to submit
    // an appropriate number of groups.
    threadGroupsX = Mathf.CeilToInt(processingTexture[0].width / 8f);
    threadGroupsY = Mathf.CeilToInt(processingTexture[0].height / 8f);
  }

  void Update() {
    // Process the shaders.
    shader.Dispatch(kernelGreyscale, threadGroupsX, threadGroupsY, 1);
    shader.Dispatch(kernelHorizSobel, threadGroupsX, threadGroupsY, 1);
    shader.Dispatch(kernelVertSobel, threadGroupsX, threadGroupsY, 1);
    shader.Dispatch(kernelCombSobel, threadGroupsX, threadGroupsY, 1);
    // Assign the threshold to the filter.
    shader.SetFloat("threshold", threshold);
    if (hardThreshold) {
      shader.Dispatch(kernelThreshold, threadGroupsX, threadGroupsY, 1);
    } else {
      shader.Dispatch(kernelThresholdFuzzy, threadGroupsX, threadGroupsY, 1);
    }
    // Colours are implicitly converted to vector4s.
    shader.SetVector("lineColour", lineColour);
    shader.Dispatch(kernelOutline, threadGroupsX, threadGroupsY, 1);

    // The two extra normalisation functions for our sobel display.
    shader.Dispatch(kernelHorizNorm, threadGroupsX, threadGroupsY, 1);
    shader.Dispatch(kernelVertNorm, threadGroupsX, threadGroupsY, 1);
  }

  public void SetThreshold(float value) {
    threshold = value;
  }

  public void SetHardThreshold(bool value) {
    hardThreshold = value;
  }

  public void Pause() {
    if (videoPlayer != null) {
      if (videoPlayer.isPlaying) {
        videoPlayer.Pause();
      } else {
        videoPlayer.Play();
      }
    }
  }

  public void IterateColours() {
    Color[] colours = new Color[] { Color.red, Color.green, Color.blue, Color.cyan, Color.magenta,
                                    Color.yellow, Color.white, Color.grey, Color.black};
    colourIdx = (colourIdx + 1) % colours.Length;
    lineColour = colours[colourIdx];
  }
}


