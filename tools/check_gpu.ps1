param(
    [string]$Build = 'build-gpu',
    [string]$Configuration = 'Release',
    [string]$MnistData = ''
)
$ErrorActionPreference = 'Stop'
& cmake -S . -B $Build -DML_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_TESTING=ON
if ($LASTEXITCODE -ne 0) { throw 'CUDA configuration failed; use an MSVC developer environment on Windows.' }
& cmake --build $Build --config $Configuration --parallel
if ($LASTEXITCODE -ne 0) { throw 'CUDA build failed.' }
# Run the hardware test directly: a CTest skip alone must not be mistaken for GPU validation.
$test = Join-Path $Build "tests\$Configuration\clarni_test_nd_cuda.exe"
if (-not (Test-Path -LiteralPath $test)) { $test = Join-Path $Build 'tests\clarni_test_nd_cuda.exe' }
& $test
if ($LASTEXITCODE -eq 77) { throw 'GPU test skipped: no usable NVIDIA device. Hardware validation was not performed.' }
if ($LASTEXITCODE -ne 0) { throw 'N-D CUDA hardware test failed.' }
& ctest --test-dir $Build -C $Configuration --output-on-failure
if ($LASTEXITCODE -ne 0) { throw 'CTest failed.' }
if ($MnistData) {
    $mnist = Join-Path $Build "examples\$Configuration\clarni_example_mnist.exe"
    if (-not (Test-Path -LiteralPath $mnist)) { $mnist = Join-Path $Build 'examples\clarni_example_mnist.exe' }
    & $mnist --data $MnistData --backend cuda --output runs\mnist-cuda-check --train-limit 300 --validation 60 --test-limit 100 --epochs 1
    if ($LASTEXITCODE -ne 0) { throw 'CUDA MNIST smoke run failed.' }
}
Write-Host 'N-D CUDA hardware checks passed.'
