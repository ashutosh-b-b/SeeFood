/*
======================================================================
components/Loader.tsx
======================================================================
A simple, reusable loading indicator component. No changes needed.
*/
import React from 'react';
import { View, ActivityIndicator, Text, StyleSheet } from 'react-native';

export default function Loader() {
  return (
    <View style={styles.container}>
      <ActivityIndicator size="large" color="#007AFF" />
      <Text style={styles.loadingText}>Analyzing food...</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#1c1c1e',
  },
  loadingText: {
    marginTop: 15,
    fontSize: 18,
    color: '#8e8e93',
  },
});
