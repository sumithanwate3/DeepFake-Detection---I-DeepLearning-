import React, { useState } from 'react';
import { getStorage, ref, uploadBytesResumable, getDownloadURL } from 'firebase/storage';
import { firebaseApp } from './FirebaseApp';
import { v4 as uuidv4 } from 'uuid';
import axios from 'axios';

const storage = getStorage(firebaseApp);

const FileUploadComponent = () => {
  const [file, setFile] = useState(null);
  const [uploadingMedia, setUploadingMedia] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [warning, setWarning] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setWarning(null); 
    setResult(null);
  };

  const handleUpload = async () => {
    if (uploadingMedia || !file) {
      setWarning('Please select a file.'); // Show warning if no file is selected
      return;
    }
  
    try {
      setUploadingMedia(true);
      const uniqueMediaId = uuidv4();
      const mediaRef = ref(storage, `media/${uniqueMediaId}`);
      const uploadTask = uploadBytesResumable(mediaRef, file);
  
      uploadTask.on('state_changed',
        (snapshot) => {
          const progress = (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
          setProgress(progress);
        },
        (error) => {
          console.error('Error uploading image to Firebase Storage:', error);
          setUploadingMedia(false);
        },
        async () => {
          try {
            const downloadURL = await getDownloadURL(uploadTask.snapshot.ref);
            console.log('File available at', downloadURL);
            const response = await axios.get(`http://127.0.0.1:8000/app/process`, { 
              params: { url: downloadURL } 
            });
            console.log('Response from backend:', response.data);
            setResult(JSON.stringify(response.data));
          } catch (error) {
            console.error('Error sending URL to backend:', error);
          } finally {
            setUploadingMedia(false);
          }
        }
      );
    } catch (error) {
      console.error('Error uploading image to Firebase Storage:', error);
      setUploadingMedia(false);
    }
  };

  return (
    <div className="relative h-screen flex flex-col items-center justify-center">
      <div className="absolute top-0 left-0 p-8">
        <h1 className="text-3xl font-bold">Deepfake Detector</h1>
      </div>
      <div className="p-8 border border-gray-300 shadow-md rounded-md flex flex-col items-center">
        <h2 className="text-2xl font-semibold mb-4">Upload Media</h2>
        <input 
          type="file" 
          onChange={handleFileChange} 
          accept="image/*,video/*" // Allow only images and videos
          className="mb-4"
        />
        <button 
          className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 mb-4 w-full"
          onClick={handleUpload}
        >
          {uploadingMedia ? 'Checking...' : 'Check'}
        </button>
        {uploadingMedia && <progress className="w-full mb-4" value={progress} max="100">{progress}%</progress>}
        {warning && <p className="text-red-500 mb-4">{warning}</p>}
        {result && 
          <div className="border border-gray-300 rounded-md p-4 w-full">
            <h3 className="text-xl font-semibold mb-2">Result</h3>
            <p className={`text-lg ${JSON.parse(result).result === 'Real' ? 'text-green-500' : 'text-red-500'}`}>
              {JSON.parse(result).result}
            </p>
          </div>
        }
      </div>
    </div>
  );
};

export default FileUploadComponent;
