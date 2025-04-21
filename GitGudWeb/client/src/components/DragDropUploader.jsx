import React, { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Button } from "@mui/material";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";

const DragDropUploader = ({ onFilesAdded }) => {
  const onDrop = useCallback((acceptedFiles) => {
    onFilesAdded(acceptedFiles);
  }, [onFilesAdded]);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    noClick: false,  
    multiple: false, 
  });

  return (
    <Button
      {...getRootProps()}
      variant="contained"
      component="label"
      startIcon={<CloudUploadIcon />}
      sx={{ display: "flex", alignItems: "center" }}
    >
      Dosya Seç
      <input {...getInputProps()} hidden accept="image/*" />
    </Button>
  );
};

export default DragDropUploader;
