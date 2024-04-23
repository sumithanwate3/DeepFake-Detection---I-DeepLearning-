import { initializeApp } from "firebase/app";

const firebaseConfig = {
  apiKey: "AIzaSyB8smg2Scg7uhNtkixJfM-LymHmDiV-hkE",
  authDomain: "deepfake-25399.firebaseapp.com",
  projectId: "deepfake-25399",
  storageBucket: "deepfake-25399.appspot.com",
  messagingSenderId: "785541095662",
  appId: "1:785541095662:web:e1ae61c7cc1aa086a553ff",
  measurementId: "G-JQQ5YDH9D0"
};

const firebaseApp = initializeApp(firebaseConfig);

export { firebaseApp};