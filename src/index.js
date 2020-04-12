
import * as bodyPix from '@tensorflow-models/body-pix';

const load = async () => {
  const preprocessImg = document.getElementById('preprocess');
  preprocessImg.style.display = 'none';

  const processed = document.getElementById('processed');
  const processedCtx = processed.getContext('2d');
  processed.width = preprocessImg.width;
  processed.height = preprocessImg.height;
  processedCtx.drawImage(preprocessImg, 0, 0);

  const example = document.getElementById('example');
  example.width = preprocessImg.width;
  example.height = preprocessImg.height;
  const exampleCtx = example.getContext('2d');

  const net = await bodyPix.load({
    architecture: 'ResNet50',
    outputStride: 32,
    quantBytes: 2,
  });

  /**
   * One of (see documentation below):
   *   - net.segmentPerson
   *   - net.segmentPersonParts
   *   - net.segmentMultiPerson
   *   - net.segmentMultiPersonParts
   * See documentation below for details on each method.
    */
  const segmentation = await net.segmentPersonParts(preprocessImg, {
    flipHorizontal: false,
    internalResolution: 'full',
    scoreThreshold: 0.2,
    segmentationThreshold: 0.5,
  });

  const {data: imgData} = processedCtx.getImageData(0, 0, example.width, example.height);
  const newImg = exampleCtx.createImageData(example.width, example.height);
  const newImgData = newImg.data;

  // Apply the effect
  for (let i = 0; i < segmentation.data.length; i++) {
    const [r, g, b, a] = [imgData[i * 4], imgData[i * 4 + 1], imgData[i * 4 + 2], imgData[i * 4 + 3]];

    [
      newImgData[i * 4],
      newImgData[i * 4 + 1],
      newImgData[i * 4 + 2],
      newImgData[i * 4 + 3],
    ] = segmentation.data[i] > 1 && segmentation.data[i] < 6 || segmentation.data[i] === 12 || segmentation.data[i] === 13 ? [r, g, b, a] : [0, 0, 0, 255];
  }

  // Draw the new image back to canvas
  exampleCtx.putImageData(newImg, 0, 0);
};

load();

