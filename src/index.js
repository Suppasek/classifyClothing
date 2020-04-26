
import * as bodyPix from '@tensorflow-models/body-pix';
import axios from 'axios';

const imageLoader = document.getElementById('imageLoader');

const handleImage = async (e) => {
  const net = await bodyPix.load({
    architecture: 'ResNet50',
    outputStride: 32,
    quantBytes: 2,
  });

  const processed = document.getElementById('processed');
  const processedCtx = processed.getContext('2d');
  const reader = new FileReader();
  const img = new Image();
  reader.onload = async (event) => {
    img.onload = async () => {
      processed.width = img.width;
      processed.height = img.height;
      processedCtx.save();
      processedCtx.globalAlpha = 0.5;
      processedCtx.drawImage(img, 0, 0);
      processedCtx.textBaseline = 'middle';
      processedCtx.textAlign = 'center';
      processedCtx.font = '30px Arial';
      processedCtx.fillText('Loading', img.width/2, img.height/2);
      // const preprocessImg = document.getElementById('preprocess');
      // preprocessImg.style.display = 'none';

      const example = document.getElementById('example');
      example.width = processed.width;
      example.height = processed.height;
      const exampleCtx = example.getContext('2d');

      /**
     * One of (see documentation below):
     *   - net.segmentPerson
     *   - net.segmentPersonParts
     *   - net.segmentMultiPerson
     *   - net.segmentMultiPersonParts
     * See documentation below for details on each method.
      */
      const segmentation = await net.segmentPersonParts(img, {
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

      example.toBlob((async (blob) => {
        const formData = new FormData();
        formData.append('img', blob);
        const result = await axios.post('http://localhost:5000', formData, {headers: {
          'Content-Type': 'multipart/form-data',
        }});
        const [x, y, w, h, classifyResult] = result.data;
        processedCtx.restore();
        processedCtx.drawImage(img, 0, 0);
        processedCtx.beginPath();
        processedCtx.rect(x, y, w, h);
        processedCtx.strokeStyle = 'red';
        processedCtx.stroke();

        document.getElementById('result').innerHTML = classifyResult;
      }), 'image/jpeg');
    };
    img.src = event.target.result;
  };


  reader.readAsDataURL(e.target.files[0]);
};

imageLoader.addEventListener('change', handleImage, false);
