import express from 'express';
import {PythonShell} from 'python-shell';
import multer from 'multer';
import cors from 'cors';

const app = express();
app.use(cors());

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, './img/');
  },
  filename: (req, file, cb) => {
    cb(null, 'segment_img.jpg');
  },
});

const upload = multer({storage: storage});

const options = {
  mode: 'text',
  pythonOptions: ['-u'], // get print results in real-time
};

app.post('/', upload.single('img'), (req, res) => {
  PythonShell.run('classify.py', options, (err, results) => {
    if (err) throw err;
    // results is an array consisting of messages collected during execution
    console.log(results);
    res.send(results);
  });
});

app.listen(5000);

