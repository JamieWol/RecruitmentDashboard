import React, { useState } from "react";
import { storage, db } from "./firebase";
import { ref, uploadBytes, getDownloadURL } from "firebase/storage";
import { collection, addDoc, serverTimestamp } from "firebase/firestore";
import { useAuth } from "./AuthContext";
import Papa from "papaparse";

export default function UploadCSV() {
  const [file, setFile] = useState(null);
  const { user } = useAuth();
  const [message, setMessage] = useState("");

  const handleUpload = async () => {
    if (!file || !user) return;
    const storageRef = ref(storage, `userCSVs/${user.uid}/${file.name}`);
    await uploadBytes(storageRef, file);
    const url = await getDownloadURL(storageRef);

    const parsed = await new Promise((resolve) => {
      Papa.parse(file, { header: true, complete: (res) => resolve(res.data) });
    });

    await addDoc(collection(db, "userCSVs"), {
      uid: user.uid,
      fileName: file.name,
      url,
      createdAt: serverTimestamp(),
      metrics: parsed
    });

    setMessage("CSV uploaded successfully!");
  };

  return (
    <div style={{ padding: 20, background: "#f0f0f0", borderRadius: 8 }}>
      <input type="file" accept=".csv" onChange={e => setFile(e.target.files[0])} />
      <button onClick={handleUpload} style={{ marginLeft: 10, padding: "5px 15px" }}>
        Upload
      </button>
      {message && <div style={{ marginTop: 10 }}>{message}</div>}
    </div>
  );
}


