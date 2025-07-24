/*
======================================================================
app/camera.tsx (Camera Screen)
======================================================================
This screen now features a "cropped" camera view, showing the camera
preview inside a square frame instead of full-screen.
*/
import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Alert } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { useRouter } from 'expo-router';

export default function CameraScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  const router = useRouter();

  useEffect(() => {
    if (!permission?.granted) {
      requestPermission();
    }
  }, [permission]);

  if (!permission) {
    return <View />;
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.permissionText}>
          We need your permission to show the camera
        </Text>
        <TouchableOpacity onPress={requestPermission} style={styles.button}>
            <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const takePicture = async () => {
    if (cameraRef.current) {
        try {
            // Take the picture but DO NOT ask for base64 here.
            // We only need the local file URI, which is much faster.
            const photo = await cameraRef.current.takePictureAsync();
            
            if (photo?.uri) {
                // Navigate instantly to the loader screen with the URI.
                router.push({
                  pathname: '/loader',
                  params: { uri: photo.uri },
                });
            } else {
                Alert.alert("Error", "Could not capture image URI.");
            }
        } catch (error) {
            console.error("Failed to take picture:", error);
            Alert.alert("Error", "Could not take a picture.");
        }
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.promptText}>Position food in the square</Text>
      
      {/* This is the frame for our camera view */}
      <View style={styles.cameraContainer}>
        <CameraView style={styles.camera} facing="back" ref={cameraRef} />
      </View>
      
      {/* The capture button is now positioned below the frame */}
      <View style={styles.bottomContainer}>
        <TouchableOpacity style={styles.captureButton} onPress={takePicture} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#000',
    },
    promptText: {
        color: 'white',
        fontSize: 18,
        fontWeight: '500',
        position: 'absolute',
        top: '15%',
    },
    // This view creates the square frame
    cameraContainer: {
        width: '85%',
        aspectRatio: 1, // This makes the height equal to the width, creating a square
        borderRadius: 20,
        overflow: 'hidden', // This "crops" the camera view to the container's bounds
        borderWidth: 2,
        borderColor: 'rgba(255, 255, 255, 0.3)',
    },
    camera: {
        flex: 1, // The CameraView will fill the square container
    },
    bottomContainer: {
        position: 'absolute',
        bottom: 50,
        width: '100%',
        alignItems: 'center',
    },
    captureButton: {
        width: 70,
        height: 70,
        borderRadius: 35,
        backgroundColor: '#fff',
        borderWidth: 5,
        borderColor: 'rgba(255, 255, 255, 0.5)',
    },
    permissionText: {
        textAlign: 'center',
        color: 'white',
        fontSize: 18,
        margin: 20,
    },
    button: {
        backgroundColor: '#007AFF',
        padding: 15,
        borderRadius: 10,
        marginTop: 20,
        alignSelf: 'center',
    },
    buttonText: {
        color: 'white',
        fontSize: 16,
        fontWeight: 'bold',
    },
});
