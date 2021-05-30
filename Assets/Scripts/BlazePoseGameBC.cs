using System.Threading;
using Cysharp.Threading.Tasks;
using TensorFlowLite;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// BlazePose form MediaPipe
/// https://github.com/google/mediapipe
/// https://viz.mediapipe.dev/demo/pose_tracking
/// </summary>
public sealed class BlazePoseGameBC : MonoBehaviour
{

    [SerializeField, FilePopup("*.tflite")] string poseDetectionModelFile = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField, FilePopup("*.tflite")] string poseLandmarkModelFile = "coco_ssd_mobilenet_quant.tflite";
    [SerializeField] RawImage cameraView = null;
    [SerializeField] RawImage debugView = null;
    [SerializeField] Canvas canvas = null;
    [SerializeField] bool useLandmarkFilter = true;
    [SerializeField] Vector3 filterVelocityScale = Vector3.one * 10;
    [SerializeField] bool runBackground;
    [SerializeField, Range(0f, 1f)] float visibilityThreshold = 0.5f;


    WebCamTexture webcamTexture;
    PoseDetect poseDetect;
    PoseLandmarkDetect poseLandmark;

    bool rthnd;

    Vector3[] rtCorners = new Vector3[4]; // just cache for GetWorldCorners
    // [SerializeField] // for debug raw data
    Vector4[] worldJoints;
    PrimitiveDraw draw;
    PoseDetect.Result poseResult;
    PoseLandmarkDetect.Result landmarkResult;
    UniTask<bool> task;
    CancellationToken cancellationToken;
    int RTScore=0, LFScore=0, BTScore=0, isMute=0;
    public Text score;
    public GameObject left0;
    public GameObject left1;
    public GameObject right0;
    public GameObject right1;
    public GameObject bonus;
    public AudioSource smallP;
    public AudioSource bigP;
    bool NeedsDetectionUpdate => poseResult == null || poseResult.score < 0.5f;

    void Start()
    {
        // Init model
        poseDetect = new PoseDetect(poseDetectionModelFile);
        poseLandmark = new PoseLandmarkDetect(poseLandmarkModelFile);

        // Init camera 
        string cameraName = WebCamUtil.FindName(new WebCamUtil.PreferSpec()
        {
            isFrontFacing = false,
            kind = WebCamKind.WideAngle,
        });
        webcamTexture = new WebCamTexture(cameraName, 1280, 720, 30);
        cameraView.texture = webcamTexture;
        webcamTexture.Play();
        Debug.Log($"Starting camera: {cameraName}");
        Debug.Log($"Screen: {(float)Screen.width } x { (float)Screen.height}");

        draw = new PrimitiveDraw(Camera.main, gameObject.layer);
        worldJoints = new Vector4[PoseLandmarkDetect.JointCount];
        left1.SetActive(false);
        right1.SetActive(false);
        bonus.SetActive(false);
        cancellationToken = this.GetCancellationTokenOnDestroy();

    }

    void OnDestroy()
    {
        webcamTexture?.Stop();
        poseDetect?.Dispose();
        poseLandmark?.Dispose();
        draw?.Dispose();
    }

    void Update()
    {
        if (runBackground)
        {
            if (task.Status.IsCompleted())
            {
                task = InvokeAsync();
            }
        }
        else
        {
            Invoke();
        }

        if (poseResult != null && poseResult.score > 0f)
        {
            // DrawFrame(poseResult);
        }

        if (landmarkResult != null && landmarkResult.score > 0.2f)
        {
            // DrawCropMatrix(poseLandmark.CropMatrix);
            DrawJoints(landmarkResult.joints);
        }
    }

    void DrawFrame(PoseDetect.Result pose)
    {
        Vector3 min = rtCorners[0];
        Vector3 max = rtCorners[2];

        draw.color = Color.green;
        draw.Rect(MathTF.Lerp(min, max, pose.rect, true), 0.02f, min.z);

        foreach (var kp in pose.keypoints)
        {
            draw.Point(MathTF.Lerp(min, max, (Vector3)kp, true), 0.05f);
        }
        draw.Apply();
    }

    void DrawCropMatrix(in Matrix4x4 matrix)
    {
        draw.color = Color.red;

        Vector3 min = rtCorners[0];
        Vector3 max = rtCorners[2];

        var mtx = WebCamUtil.GetMatrix(-webcamTexture.videoRotationAngle, false, webcamTexture.videoVerticallyMirrored)
            * matrix.inverse;
        Vector3 a = MathTF.LerpUnclamped(min, max, mtx.MultiplyPoint3x4(new Vector3(0, 0, 0)));
        Vector3 b = MathTF.LerpUnclamped(min, max, mtx.MultiplyPoint3x4(new Vector3(1, 0, 0)));
        Vector3 c = MathTF.LerpUnclamped(min, max, mtx.MultiplyPoint3x4(new Vector3(1, 1, 0)));
        Vector3 d = MathTF.LerpUnclamped(min, max, mtx.MultiplyPoint3x4(new Vector3(0, 1, 0)));

        draw.Quad(a, b, c, d, 0.02f);
        draw.Apply();
    }

    void DrawJoints(Vector4[] joints)
    {
        draw.color = Color.white;

        // draw.Point(new Vector3((float)375/100,0,0),(float)0.75);
        // draw.Point(new Vector3((float)-525/100,0,0),(float)0.75);

        // Vector3 min = rtCorners[0];
        // Vector3 max = rtCorners[2];
        // Debug.Log($"rtCorners min: {min}, max: {max}");

        // Apply webcam rotation to draw landmarks correctly
        Matrix4x4 mtx = WebCamUtil.GetMatrix(-webcamTexture.videoRotationAngle, false, webcamTexture.videoVerticallyMirrored);

        // float zScale = (max.x - min.x) / 2;
        float zScale = 1;
        float zOffset = canvas.planeDistance;
        float aspect = (float)Screen.width / (float)Screen.height;
        Vector3 scale, offset;
        if (aspect > 1)
        {
            scale = new Vector3(1f / aspect, 1f, zScale);
            offset = new Vector3((1 - 1f / aspect) / 2, 0, zOffset);
        }
        else
        {
            scale = new Vector3(1f, aspect, zScale);
            offset = new Vector3(0, (1 - aspect) / 2, zOffset);
        }
        // Debug.Log($"Current Joints: {joints.Length}");
        // Update world joints
        var camera = canvas.worldCamera;
        for (int i = 0; i < joints.Length; i++)
        {
            Vector3 p = mtx.MultiplyPoint3x4((Vector3)joints[i]);
            p = Vector3.Scale(p, scale);
            // p = new Vector3((-1)*p.x, p.y, p.z);
            p = p + offset;
            p = camera.ViewportToWorldPoint(p);

            // w is visibility
            // worldJoints[i] = new Vector4((24/10)*p.x, (24/10)*p.y, p.z, joints[i].w);
            worldJoints[i] = new Vector4(p.x, p.y, p.z, joints[i].w);
            if(i==21){
                
                
            }
        }

        // Draw
        for (int i = 0; i < worldJoints.Length; i++)
        {
            Vector4 p = worldJoints[i];
            if (p.w > visibilityThreshold)
            {
                // draw.Cube(p, 0.2f);
            }
        }
        var connections = PoseLandmarkDetect.Connections;
        for (int i = 0; i < connections.Length; i += 2)
        {
            var a = worldJoints[connections[i]];
            var b = worldJoints[connections[i + 1]];
            if (a.w > visibilityThreshold || b.w > visibilityThreshold)
            {
                // draw.Line3D(a, b, 0.05f);
            }
        }
        
        if(RightCheck(worldJoints[16]) || RightCheck(worldJoints[22]) ||RightCheck(worldJoints[18]) ||RightCheck(worldJoints[20])){
            draw.color=Color.red;
            rthnd=true;
            RTScore++;
            right1.SetActive(true);
            right0.SetActive(false);
        }
        else{
            RTScore=0;
            rthnd=false;
            right1.SetActive(false);
            right0.SetActive(true);
        }
        
        if(LeftCheck(worldJoints[21]) || LeftCheck(worldJoints[15]) || LeftCheck(worldJoints[17]) || LeftCheck(worldJoints[19])){
            draw.color=Color.blue;
            if(rthnd==true){
                draw.color=Color.yellow;
                BTScore++;
                bonus.SetActive(true);
            }
        
            left1.SetActive(true);
            left0.SetActive(false);
            LFScore++; 
        }
        else{
            BTScore=0;
            bonus.SetActive(false);
            left1.SetActive(false);
            left0.SetActive(true);
            LFScore=0;
        }

        if(RTScore>=150){
            RTScore=0;
            ScorePlus(5);
            if(isMute==0){smallP.Play();}
        }
        if(LFScore>=150){
            LFScore=0;
            ScorePlus(5);
            if(isMute==0){smallP.Play();}
        }
        if(BTScore>=150){
            BTScore=0;
            ScorePlus(10);
            if(isMute==0){bigP.Play();}
        }
        Debug.Log($"Left: {LFScore}");
        Debug.Log($"Right: {RTScore}");
        Debug.Log($"Both: {BTScore}");
        draw.Apply();
    }

    bool RightCheck(Vector4 p){
        if(p.x>3.3 && p.x<4.1){
            if(p.y>-0.37 && p.y<0.37){
                return true;
            }
        }
        return false;
    }

    bool LeftCheck(Vector4 p){
        if(p.x>-5.6 && p.x<-4.9){
            if(p.y>-0.37 && p.y<0.37){
                return true;
            }
        }
        return false;
    }

    void Invoke()
    {
        if (NeedsDetectionUpdate)
        {
            poseDetect.Invoke(webcamTexture);
            cameraView.material = poseDetect.transformMat;
            cameraView.rectTransform.GetWorldCorners(rtCorners);
            poseResult = poseDetect.GetResults(0.7f, 0.3f);
        }
        if (poseResult.score < 0)
        {
            poseResult = null;
            landmarkResult = null;
            return;
        }
        poseLandmark.Invoke(webcamTexture, poseResult);
        debugView.texture = poseLandmark.inputTex;

        if (useLandmarkFilter)
        {
            poseLandmark.FilterVelocityScale = filterVelocityScale;
        }
        landmarkResult = poseLandmark.GetResult(useLandmarkFilter);

        if (landmarkResult.score < 0.3f)
        {
            poseResult.score = landmarkResult.score;
        }
        else
        {
            poseResult = PoseLandmarkDetect.LandmarkToDetection(landmarkResult);
        }
    }

    async UniTask<bool> InvokeAsync()
    {
        if (NeedsDetectionUpdate)
        {
            // Note: `await` changes PlayerLoopTiming from Update to FixedUpdate.
            poseResult = await poseDetect.InvokeAsync(webcamTexture, cancellationToken, PlayerLoopTiming.FixedUpdate);
        }
        if (poseResult.score < 0)
        {
            poseResult = null;
            landmarkResult = null;
            return false;
        }

        if (useLandmarkFilter)
        {
            poseLandmark.FilterVelocityScale = filterVelocityScale;
        }
        landmarkResult = await poseLandmark.InvokeAsync(webcamTexture, poseResult, useLandmarkFilter, cancellationToken, PlayerLoopTiming.Update);

        // Back to the update timing from now on 
        if (cameraView != null)
        {
            cameraView.material = poseDetect.transformMat;
            cameraView.rectTransform.GetWorldCorners(rtCorners);
        }
        if (debugView != null)
        {
            debugView.texture = poseLandmark.inputTex;
        }

        // Generate poseResult from landmarkResult
        if (landmarkResult.score < 0.3f)
        {
            poseResult.score = landmarkResult.score;
        }
        else
        {
            poseResult = PoseLandmarkDetect.LandmarkToDetection(landmarkResult);
        }

        return true;
    }

    void ScorePlus(int AddScore){
        int curr;
        int.TryParse(score.text.ToString(), out curr);
        curr=curr + AddScore;
        score.text = curr.ToString();
    }

    public void muteR(){
        if(isMute==0){
            isMute=1;
        }
        else{
            isMute=0;
        }
    }
}
