/*
======================================================================
app/camera.tsx (Camera Screen)
======================================================================
This screen now only takes a picture and gets its local file URI,
then immediately navigates to the new loader screen.
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
      <CameraView style={styles.camera} facing="back" ref={cameraRef}>
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.captureButton} onPress={takePicture} />
        </View>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        backgroundColor: '#000',
    },
    camera: {
        flex: 1,
    },
    permissionText: {
        textAlign: 'center',
        color: 'white',
        fontSize: 18,
        margin: 20,
    },
    buttonContainer: {
        flex: 1,
        backgroundColor: 'transparent',
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'flex-end',
        marginBottom: 40,
    },
    captureButton: {
        width: 70,
        height: 70,
        borderRadius: 35,
        backgroundColor: '#fff',
        borderWidth: 5,
        borderColor: 'rgba(255, 255, 255, 0.5)',
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
