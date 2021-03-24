/*
See LICENSE folder for this sample’s licensing information.

Abstract:
The host app renderer.
*/

import Metal
import MetalKit
import ARKit

final class Renderer {
    // Maximum number of points we store in the point cloud
    private let maxPoints = 500_000
    // Number of sample points on the grid
    private let numGridPoints = 500
    // Particle's size in pixels
    // 보여줄 때 point의 사이즈를 변경할 수 있음.
    private let particleSize: Float = 10
    // We only use landscape orientation in this app
    private let orientation = UIInterfaceOrientation.landscapeRight
    // Camera's threshold values for detecting when the camera moves so that we can accumulate the points
    private let cameraRotationThreshold = cos(2 * .degreesToRadian)
    private let cameraTranslationThreshold: Float = pow(0.02, 2)   // (meter-squared)
    // The max number of command buffers in flight
    private let maxInFlightBuffers = 3
    
    private lazy var rotateToARCamera = Self.makeRotateToARCameraMatrix(orientation: orientation)
    private let session: ARSession
    
    // Metal objects and textures
    private let device: MTLDevice
    private let library: MTLLibrary
    private let renderDestination: RenderDestinationProvider
    private let relaxedStencilState: MTLDepthStencilState
    private let depthStencilState: MTLDepthStencilState
    private let commandQueue: MTLCommandQueue
    private lazy var unprojectPipelineState = makeUnprojectionPipelineState()!
    private lazy var rgbPipelineState = makeRGBPipelineState()!
    private lazy var particlePipelineState = makeParticlePipelineState()!
    // texture cache for captured image
    private lazy var textureCache = makeTextureCache()
    private var capturedImageTextureY: CVMetalTexture?
    private var capturedImageTextureCbCr: CVMetalTexture?
    private var depthTexture: CVMetalTexture?
    private var confidenceTexture: CVMetalTexture?
    
    // Multi-buffer rendering pipeline
    private let inFlightSemaphore: DispatchSemaphore
    private var currentBufferIndex = 0
    
    // The current viewport size
    private var viewportSize = CGSize()
    // The grid of sample points
    private lazy var gridPointsBuffer = MetalBuffer<Float2>(device: device,
                                                            array: makeGridPoints(),
                                                            index: kGridPoints.rawValue, options: [])
    
    
    
    //depth map, rgb map 두개 뽑는 코드.
    //
    func pixelToImage(pixelBuffer: CVPixelBuffer) -> UIImage? {
            let context = CIContext()
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            if let image = context.createCGImage(ciImage, from: ciImage.extent) {
                return UIImage(cgImage: image)
            }
            return nil
    }

    func depthImage(frame: ARFrame) -> CVPixelBuffer {
        //sceneDepth.depthMap >> depthMap
        let depthMap = frame.sceneDepth!.depthMap
        return depthMap
        }

    func documentDirectoryPath() -> URL? {
        let path = FileManager.default.urls(for: .documentDirectory,
                                            in: .userDomainMask)
        return path.first
    }
    
    func saveJpg() {

        var sampleFrame: ARFrame { session.currentFrame! }
        let dep = depthImage(frame: sampleFrame)
        let rgb = sampleFrame.capturedImage
        let imgdep = pixelToImage(pixelBuffer: dep)
        let imgrgb = pixelToImage(pixelBuffer: rgb)
        if let depData = imgdep?.jpegData(compressionQuality: 0.5),
           let path = documentDirectoryPath()?.appendingPathComponent("Depth_\(UUID().uuidString).jpg") {
            try? depData.write(to: path)
        }
        if let rgbData = imgrgb?.jpegData(compressionQuality: 0.5),
           let path = documentDirectoryPath()?.appendingPathComponent("RGB_\(UUID().uuidString).jpg") {
            try? rgbData.write(to: path)
        }
    }

    //
    
// raw point cloud 저장하는 코드
    
    var isSavingFile = false

    func savePointsToFile() {
      guard !self.isSavingFile else { return }
      self.isSavingFile = true

        // 1
        var fileToWrite = ""
//        let headers = ["ply", "format ascii 1.0", "element vertex \(currentPointCount)", "property float x", "property float y", "property float z", "property float cx", "property float cy", "property float depth", "property uchar alpha", "element face 0", "property list uchar int vertex_indices", "end_header"]
        let headers = ["ply", "format ascii 1.0", "element vertex \(49152)", "property float x", "property float y", "property float z", "property uchar red", "property uchar green", "property uchar blue", "property float cx", "property float cy", "property float depth", "property uchar alpha", "element face 0", "property list uchar int vertex_indices", "end_header"]
        for header in headers {
            fileToWrite += header
            fileToWrite += "\r\n"
        }
        
        var sampleFrame: ARFrame { session.currentFrame! }
        let dep = depthImage(frame: sampleFrame)
        
        let cI = pointCloudUniformsBuffers[currentBufferIndex][0].cameraIntrinsicsInversed
        let lT = pointCloudUniformsBuffers[currentBufferIndex][0].localToWorld
        
        CVPixelBufferLockBaseAddress(dep, CVPixelBufferLockFlags.readOnly)
        let byteAddress = CVPixelBufferGetBaseAddress(dep)
        
        //unsafeBitCast: instance의 type을 변경해줌.
        //depthMap은 CVPixelBuffer type이므로 곧바로 픽셀 값 확인이 어렵다, float type의 array 형태로 접근 가능하다.
        let DM = unsafeBitCast(byteAddress,to: UnsafeMutablePointer<Float32>.self)
        
//        func CVPixelBufferGetPlaneCount(_ pixelBuffer: CVPixelBuffer) -> Int ,,,, plane count
        let DepthWidth = CVPixelBufferGetWidth(dep)
        let DepthHeight = CVPixelBufferGetHeight(dep)
        
        //
        let rgb = sampleFrame.capturedImage
        let imgdep = pixelToImage(pixelBuffer: dep)
        let imgrgb = pixelToImage(pixelBuffer: rgb)
        if let depData = imgdep?.jpegData(compressionQuality: 0.5),
           let path = documentDirectoryPath()?.appendingPathComponent("Depth_\(UUID().uuidString).jpg") {
            try? depData.write(to: path)
        }
        if let rgbData = imgrgb?.jpegData(compressionQuality: 0.5),
           let path = documentDirectoryPath()?.appendingPathComponent("RGB_\(UUID().uuidString).jpg") {
            try? rgbData.write(to: path)
        }
        //
        
        for x in 0..<DepthWidth
                        {
                for y in 0..<DepthHeight
                            {
                    let depth = DM[x + y * DepthWidth]
                    let coor = Float2(Float(x), Float(y))
                    let lp = cI * simd_float3(coor, 1) * depth
                    let wp = lT * simd_float4(lp, 1)
                    let finn = wp / wp.w

                    let pvValue = "\(finn.x) \(finn.y) \(finn.z) \(Int(0)) \(Int(0)) \(Int(0)) \(coor.x) \(coor.y) \(depth)  255"

                    fileToWrite += pvValue
                    fileToWrite += "\r\n"
                }}
//        // 2
//        for i in 0..<currentPointCount {
//
//            // 3
//            let point = particlesBuffer[i]
//            let colors = point.color
//
//            // 4
//            let red = colors.x * 255.0
//            let green = colors.y * 255.0
//            let blue = colors.z * 255.0
//
//            // 5
//            let pvValue = "\(point.position.x) \(point.position.y) \(point.position.z) \(Int(red)) \(Int(green)) \(Int(blue)) \(point.texCoord.x) \(point.texCoord.y) \(point.depth)  255"
//            fileToWrite += pvValue
//            fileToWrite += "\r\n"
//        }
//        // 6
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        let documentsDirectory = paths[0]
        let file = documentsDirectory.appendingPathComponent("ply_\(UUID().uuidString).ply")

        do {

            // 7
            try fileToWrite.write(to: file, atomically: true, encoding: String.Encoding.ascii)
            self.isSavingFile = false
        } catch {
            print("Failed to write PLY file", error)
        }
    }
    
    // RGB buffer
    private lazy var rgbUniforms: RGBUniforms = {
        var uniforms = RGBUniforms()
        uniforms.radius = rgbRadius
        uniforms.viewToCamera.copy(from: viewToCamera)
        uniforms.viewRatio = Float(viewportSize.width / viewportSize.height)
        return uniforms
    }()
    private var rgbUniformsBuffers = [MetalBuffer<RGBUniforms>]()
    
    // Point Cloud buffer
    private lazy var pointCloudUniforms: PointCloudUniforms = {
        var uniforms = PointCloudUniforms()
        uniforms.maxPoints = Int32(maxPoints)
        uniforms.confidenceThreshold = Int32(confidenceThreshold)
        uniforms.particleSize = particleSize
        uniforms.cameraResolution = cameraResolution
        return uniforms
    }()
    
    private var pointCloudUniformsBuffers = [MetalBuffer<PointCloudUniforms>]()
    // Particles buffer
    private var particlesBuffer: MetalBuffer<ParticleUniforms>
    private var currentPointIndex = 0
    private var currentPointCount = 0
    
    // Camera data
    private var sampleFrame: ARFrame { session.currentFrame! }
    private lazy var cameraResolution = Float2(Float(sampleFrame.camera.imageResolution.width), Float(sampleFrame.camera.imageResolution.height))
    private lazy var viewToCamera = sampleFrame.displayTransform(for: orientation, viewportSize: viewportSize).inverted()
    private lazy var lastCameraTransform = sampleFrame.camera.transform
    
    // interfaces
    var confidenceThreshold = 1 {
        didSet {
            // apply the change for the shader
            pointCloudUniforms.confidenceThreshold = Int32(confidenceThreshold)
        }
    }
    
    var rgbRadius: Float = 0 {
        didSet {
            // apply the change for the shader
            rgbUniforms.radius = rgbRadius
        }
    }
    
    init(session: ARSession, metalDevice device: MTLDevice, renderDestination: RenderDestinationProvider) {
        self.session = session
        self.device = device
        self.renderDestination = renderDestination
        
        library = device.makeDefaultLibrary()!
        commandQueue = device.makeCommandQueue()!
        
        // initialize our buffers
        for _ in 0 ..< maxInFlightBuffers {
            rgbUniformsBuffers.append(.init(device: device, count: 1, index: 0))
            pointCloudUniformsBuffers.append(.init(device: device, count: 1, index: kPointCloudUniforms.rawValue))
        }
        particlesBuffer = .init(device: device, count: maxPoints, index: kParticleUniforms.rawValue)
        
        // rbg does not need to read/write depth
        let relaxedStateDescriptor = MTLDepthStencilDescriptor()
        relaxedStencilState = device.makeDepthStencilState(descriptor: relaxedStateDescriptor)!
        
        // setup depth test for point cloud
        let depthStateDescriptor = MTLDepthStencilDescriptor()
        depthStateDescriptor.depthCompareFunction = .lessEqual
        depthStateDescriptor.isDepthWriteEnabled = true
        depthStencilState = device.makeDepthStencilState(descriptor: depthStateDescriptor)!
        
        inFlightSemaphore = DispatchSemaphore(value: maxInFlightBuffers)
    }
    
    func drawRectResized(size: CGSize) {
        viewportSize = size
    }
   
    private func updateCapturedImageTextures(frame: ARFrame) {
        // Create two textures (Y and CbCr) from the provided frame's captured image
        let pixelBuffer = frame.capturedImage
        //Y and CbCr map 2개 생성, plane 0 1920*1440, plane 1 960*720
//        print ("RGB: ",  frame.capturedImage)
        guard CVPixelBufferGetPlaneCount(pixelBuffer) >= 2 else {
            return
        }
        
        capturedImageTextureY = makeTexture(fromPixelBuffer: pixelBuffer, pixelFormat: .r8Unorm, planeIndex: 0)
        capturedImageTextureCbCr = makeTexture(fromPixelBuffer: pixelBuffer, pixelFormat: .rg8Unorm, planeIndex: 1)
    }
    
    private func updateDepthTextures(frame: ARFrame) -> Bool {
        
        //sceneDepth.depthMap >> depthMap
        guard let depthMap = frame.sceneDepth?.depthMap,
            let confidenceMap = frame.sceneDepth?.confidenceMap else {
                return false
        }
//        unowned(unsafe) open var depthMap: CVPixelBuffer { get }
        //Captured image의 width height
        //depthmap,confidencemap의 width height
        depthTexture = makeTexture(fromPixelBuffer: depthMap, pixelFormat: .r32Float, planeIndex: 0)
        confidenceTexture = makeTexture(fromPixelBuffer: confidenceMap, pixelFormat: .r8Uint, planeIndex: 0)
    
        CVPixelBufferLockBaseAddress(depthMap, CVPixelBufferLockFlags.readOnly)
        let byteAddress = CVPixelBufferGetBaseAddress(depthMap)
        
        //unsafeBitCast: instance의 type을 변경해줌.
        //depthMap은 CVPixelBuffer type이므로 곧바로 픽셀 값 확인이 어렵다, float type의 array 형태로 접근 가능하다.
        let floatBuffer = unsafeBitCast(byteAddress,to: UnsafeMutablePointer<Float32>.self)
        
//        func CVPixelBufferGetPlaneCount(_ pixelBuffer: CVPixelBuffer) -> Int ,,,, plane count
        let countdepth = CVPixelBufferGetWidth(depthMap)
        dump(countdepth)
        
        //depthMap, confidenceMap 의 사이즈 check
//        print ("Depth: ",  frame.sceneDepth!.depthMap)
//        print ("confidenceMap: ",  frame.sceneDepth!.confidenceMap.debugDescription)
        
        //1번째 depth를 출력
        dump(floatBuffer[0])
        //중간 depth를 출력
        dump(floatBuffer[256*96 + 128])
        
        
        return true
    }
    
//    var depthValues = depthTexture.GetPixels().Select(x => x.r).ToArray();
//
//                for (int x = 0; x < DepthWidth; x++)
//                {
//                    for (int y = 0; y < DepthHeight; y++)
//                    {
//                            var colX = Mathf.RoundToInt((float)x * _camTexture2D.width / DepthWidth);
//                            var colY = Mathf.RoundToInt((float)y * _camTexture2D.height / DepthHeight);
//
//                            var pixelX = Mathf.RoundToInt((float)x * _mainCam.pixelWidth / DepthWidth);
//                            var pixelY = Mathf.RoundToInt((float)y * _mainCam.pixelHeight / DepthHeight);
//
//                            var depth = depthValues[x + y * DepthWidth];
//
//                            var scrToWorld = _mainCam.ScreenToWorldPoint(new Vector3(pixelX, pixelY, depth));
//
//                            _colors.Add(_camTexture2D.GetPixel(colX, colY));
//                            _vertices.Add(scrToWorld);
//                    }
//                }
    
    private func update(frame: ARFrame) {
        // frame dependent info
        let camera = frame.camera
        let cameraIntrinsicsInversed = camera.intrinsics.inverse
        let viewMatrix = camera.viewMatrix(for: orientation)
        let viewMatrixInversed = viewMatrix.inverse
        let projectionMatrix = camera.projectionMatrix(for: orientation, viewportSize: viewportSize, zNear: 0.001, zFar: 0)
        pointCloudUniforms.viewProjectionMatrix = projectionMatrix * viewMatrix
        pointCloudUniforms.localToWorld = viewMatrixInversed * rotateToARCamera
        pointCloudUniforms.cameraIntrinsicsInversed = cameraIntrinsicsInversed
    }
    
    //이부분을 봐야할듯
    func draw() {
        guard let currentFrame = session.currentFrame,
            let renderDescriptor = renderDestination.currentRenderPassDescriptor,
            let commandBuffer = commandQueue.makeCommandBuffer(),
            let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderDescriptor) else {
                return
        }
        
        
        _ = inFlightSemaphore.wait(timeout: DispatchTime.distantFuture)
        commandBuffer.addCompletedHandler { [weak self] commandBuffer in
            if let self = self {
                self.inFlightSemaphore.signal()
            }
        }
        
//        print("command: ",commandBuffer)
        
        // update frame data
        update(frame: currentFrame)
        updateCapturedImageTextures(frame: currentFrame)
        
        // handle buffer rotating
        currentBufferIndex = (currentBufferIndex + 1) % maxInFlightBuffers
        pointCloudUniformsBuffers[currentBufferIndex][0] = pointCloudUniforms
        
//        print(pointCloudUniforms)
        
        if shouldAccumulate(frame: currentFrame), updateDepthTextures(frame: currentFrame) {
            accumulatePoints(frame: currentFrame, commandBuffer: commandBuffer, renderEncoder: renderEncoder)
        }
        
        // check and render rgb camera image
        if rgbUniforms.radius > 0 {
            var retainingTextures = [capturedImageTextureY, capturedImageTextureCbCr]
            commandBuffer.addCompletedHandler { buffer in
                retainingTextures.removeAll()
            }
            rgbUniformsBuffers[currentBufferIndex][0] = rgbUniforms
            
            renderEncoder.setDepthStencilState(relaxedStencilState)
            renderEncoder.setRenderPipelineState(rgbPipelineState)
            renderEncoder.setVertexBuffer(rgbUniformsBuffers[currentBufferIndex])
            renderEncoder.setFragmentBuffer(rgbUniformsBuffers[currentBufferIndex])
            renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(capturedImageTextureY!), index: Int(kTextureY.rawValue))
            renderEncoder.setFragmentTexture(CVMetalTextureGetTexture(capturedImageTextureCbCr!), index: Int(kTextureCbCr.rawValue))
            renderEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
       
        // render particles
        renderEncoder.setDepthStencilState(depthStencilState)
        renderEncoder.setRenderPipelineState(particlePipelineState)
        renderEncoder.setVertexBuffer(pointCloudUniformsBuffers[currentBufferIndex])
        renderEncoder.setVertexBuffer(particlesBuffer)
        //Buffer는 row한 value를 가지고 있다.
        //Texture에 좌표값이 있다.
        
        //size가 color buffer랑 point buffer랑 다르다.
//        print("Y: ",capturedImageTextureY!)
//        print("CbCr: ",capturedImageTextureCbCr!)
//        print("depth: ",depthTexture!)
//        print("confidence: ",confidenceTexture!)
        //결국 coordinate나 rgb값은 particlesBuffer에 담겨있다.
        //1920,1440
        print("point: ",pointCloudUniformsBuffers[currentBufferIndex][0].cameraResolution)
        print ("point: ",pointCloudUniformsBuffers[currentBufferIndex][0].cameraIntrinsicsInversed)
        print ("point: ",pointCloudUniformsBuffers[currentBufferIndex][0].localToWorld)
        print("grid_point: ",makeGridPoints()[0])
        print("index: ",currentBufferIndex)
//        print("point1000: ",particlesBuffer[100].gridPoint.x)
//        print("point1000: ",particlesBuffer[100].gridPoint.y)
//        print("resolution: ",pointCloudUniformsBuffers[currentBufferIndex][0])
        
//        print("point: ",particlesBuffer[0].color)
        //viewToCamera, viewRation 등이 담겨있음.
//        print("RGB: ",rgbUniformsBuffers[currentBufferIndex][0])
        //viewprojectionmatrix, ...,cameraIntrinsicsInversed matrix 등 projection 관련 값들
//        print("pointbuffer: ",pointCloudUniformsBuffers[currentBufferIndex])
//        print("Vertext: ",renderEncoder)
//        print("grid: ",gridPointsBuffer)
        
        renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: currentPointCount)
        renderEncoder.endEncoding()
        
    
            
        commandBuffer.present(renderDestination.currentDrawable!)
//        print("darwable객체 :",renderDestination.currentDrawable!)
        commandBuffer.commit()
    }
    
    private func shouldAccumulate(frame: ARFrame) -> Bool {
        let cameraTransform = frame.camera.transform
        return currentPointCount == 0
            || dot(cameraTransform.columns.2, lastCameraTransform.columns.2) <= cameraRotationThreshold
            || distance_squared(cameraTransform.columns.3, lastCameraTransform.columns.3) >= cameraTranslationThreshold
    }
    
    //point
    private func accumulatePoints(frame: ARFrame, commandBuffer: MTLCommandBuffer, renderEncoder: MTLRenderCommandEncoder) {
        pointCloudUniforms.pointCloudCurrentIndex = Int32(currentPointIndex)
        //4가지 testure를 결합해줌.
        //사이즈가 각기 다름. 어떻게 matching시키고, matching 된 각각의 데이터를 뽑아내야함.
        var retainingTextures = [capturedImageTextureY, capturedImageTextureCbCr, depthTexture, confidenceTexture]
//        print("texture: ",retainingTextures)
        //texture 사이즈가 각기 다르다, 어디서 사이즈를 맞춰주는지? matching을 시키고 visualize를 시키는지 확인하자
        //Y와 CbCr의 사이즈는 같아짐.
//        print("Y: ",capturedImageTextureY!)
//        print("CbCr: ",capturedImageTextureCbCr!)
//        print("depth: ",depthTexture!)
//        print("confidence: ",confidenceTexture!)
        
        commandBuffer.addCompletedHandler { buffer in
            retainingTextures.removeAll()
        }
//        print("texture2: ", commandBuffer)
        
        renderEncoder.setDepthStencilState(relaxedStencilState)
        renderEncoder.setRenderPipelineState(unprojectPipelineState)
        renderEncoder.setVertexBuffer(pointCloudUniformsBuffers[currentBufferIndex])
        renderEncoder.setVertexBuffer(particlesBuffer)
        renderEncoder.setVertexBuffer(gridPointsBuffer)
//        print("gridpoint: ",gridPointsBuffer)
//        print("point: ",gridPointsBuffer)
//        print("point: ",gridPointsBuffer[0].color)
        renderEncoder.setVertexTexture(CVMetalTextureGetTexture(capturedImageTextureY!), index: Int(kTextureY.rawValue))
        renderEncoder.setVertexTexture(CVMetalTextureGetTexture(capturedImageTextureCbCr!), index: Int(kTextureCbCr.rawValue))
        renderEncoder.setVertexTexture(CVMetalTextureGetTexture(depthTexture!), index: Int(kTextureDepth.rawValue))
        renderEncoder.setVertexTexture(CVMetalTextureGetTexture(confidenceTexture!), index: Int(kTextureConfidence.rawValue))
        renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: gridPointsBuffer.count)
//        print("render: ",renderEncoder)
        currentPointIndex = (currentPointIndex + gridPointsBuffer.count) % maxPoints
        currentPointCount = min(currentPointCount + gridPointsBuffer.count, maxPoints)
//        print("pointindex: ",currentPointIndex)
//        print("pointcount: ",currentPointCount)
        lastCameraTransform = frame.camera.transform
        
    }
    
//    func pointcoord(){
//        var sampleFrame: ARFrame { session.currentFrame! }
//        let dep = depthImage(frame: sampleFrame)
//
//        let cI = pointCloudUniformsBuffers[currentBufferIndex][0].cameraIntrinsicsInversed
//        let lT = pointCloudUniformsBuffers[currentBufferIndex][0].localToWorld
//
//        CVPixelBufferLockBaseAddress(dep, CVPixelBufferLockFlags.readOnly)
//        let byteAddress = CVPixelBufferGetBaseAddress(dep)
//
//        //unsafeBitCast: instance의 type을 변경해줌.
//        //depthMap은 CVPixelBuffer type이므로 곧바로 픽셀 값 확인이 어렵다, float type의 array 형태로 접근 가능하다.
//        let DM = unsafeBitCast(byteAddress,to: UnsafeMutablePointer<Float32>.self)
//
////        func CVPixelBufferGetPlaneCount(_ pixelBuffer: CVPixelBuffer) -> Int ,,,, plane count
//        let DepthWidth = CVPixelBufferGetWidth(dep)
//        let DepthHeight = CVPixelBufferGetHeight(dep)
//        for x in 0..<DepthWidth
//                        {
//                for y in 0..<DepthHeight
//                            {
//                    let depth = DM[x + y * DepthWidth]
//                    let coor = Float2(Float(x), Float(y))
//                    let lp = cI * simd_float3(coor, 1) * depth
//                    let wp = lT * simd_float4(lp, 1)
//                    let finn = wp / wp.w
//
//                }}}
}

// MARK: - Metal Helpers
//속성을 설정한다. vertex,fragment 함수 및 컬러의 첨부 속성 등 설정
private extension Renderer {
    func makeUnprojectionPipelineState() -> MTLRenderPipelineState? {
        guard let vertexFunction = library.makeFunction(name: "unprojectVertex") else {
                return nil
        }
        
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.isRasterizationEnabled = false
        descriptor.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat
        descriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        
        return try? device.makeRenderPipelineState(descriptor: descriptor)
    }
    
    func makeRGBPipelineState() -> MTLRenderPipelineState? {
        guard let vertexFunction = library.makeFunction(name: "rgbVertex"),
            let fragmentFunction = library.makeFunction(name: "rgbFragment") else {
                return nil
        }
        
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat
        descriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        
        return try? device.makeRenderPipelineState(descriptor: descriptor)
    }
    
    func makeParticlePipelineState() -> MTLRenderPipelineState? {
        guard let vertexFunction = library.makeFunction(name: "particleVertex"),
            let fragmentFunction = library.makeFunction(name: "particleFragment") else {
                return nil
        }
        
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = vertexFunction
        descriptor.fragmentFunction = fragmentFunction
        descriptor.depthAttachmentPixelFormat = renderDestination.depthStencilPixelFormat
        descriptor.colorAttachments[0].pixelFormat = renderDestination.colorPixelFormat
        descriptor.colorAttachments[0].isBlendingEnabled = true
        descriptor.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        descriptor.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        descriptor.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        
        return try? device.makeRenderPipelineState(descriptor: descriptor)
    }
    
    /// Makes sample points on camera image, also precompute the anchor point for animation
    func makeGridPoints() -> [Float2] {
        let gridArea = cameraResolution.x * cameraResolution.y
        let spacing = sqrt(gridArea / Float(numGridPoints))
        let deltaX = Int(round(cameraResolution.x / spacing))
        let deltaY = Int(round(cameraResolution.y / spacing))
        
        var points = [Float2]()
        for gridY in 0 ..< deltaY {
            let alternatingOffsetX = Float(gridY % 2) * spacing / 2
            for gridX in 0 ..< deltaX {
                let cameraPoint = Float2(alternatingOffsetX + (Float(gridX) + 0.5) * spacing, (Float(gridY) + 0.5) * spacing)
                
                points.append(cameraPoint)
            }
        }
        //points 좌표를 이야기하는 것 같음.
//        print("points: ",points)
        return points
    }
    
    func makeTextureCache() -> CVMetalTextureCache {
        // Create captured image texture cache
        var cache: CVMetalTextureCache!
        CVMetalTextureCacheCreate(nil, nil, device, nil, &cache)
        
        return cache
    }
    
    func makeTexture(fromPixelBuffer pixelBuffer: CVPixelBuffer, pixelFormat: MTLPixelFormat, planeIndex: Int) -> CVMetalTexture? {
        let width = CVPixelBufferGetWidthOfPlane(pixelBuffer, planeIndex)
        let height = CVPixelBufferGetHeightOfPlane(pixelBuffer, planeIndex)
        
        var texture: CVMetalTexture? = nil
        let status = CVMetalTextureCacheCreateTextureFromImage(nil, textureCache, pixelBuffer, nil, pixelFormat, width, height, planeIndex, &texture)
        
        if status != kCVReturnSuccess {
            texture = nil
        }

        return texture
    }
    
    static func cameraToDisplayRotation(orientation: UIInterfaceOrientation) -> Int {
        switch orientation {
        case .landscapeLeft:
            return 180
        case .portrait:
            return 90
        case .portraitUpsideDown:
            return -90
        default:
            return 0
        }
    }
    
    static func makeRotateToARCameraMatrix(orientation: UIInterfaceOrientation) -> matrix_float4x4 {
        // flip to ARKit Camera's coordinate
        let flipYZ = matrix_float4x4(
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1] )

        let rotationAngle = Float(cameraToDisplayRotation(orientation: orientation)) * .degreesToRadian
        return flipYZ * matrix_float4x4(simd_quaternion(rotationAngle, Float3(0, 0, 1)))
    }
}
