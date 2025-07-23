/*
======================================================================
app/_layout.tsx (Root Layout)
======================================================================
This is the main layout file for the app. It uses Expo Router's
Stack navigator to define the overall navigation structure.
*/
import { Stack } from 'expo-router';
import React from 'react';

export default function RootLayout() {
  return (
    <Stack>
      {/* The 'index' screen is our splash screen */}
      <Stack.Screen name="index" options={{ headerShown: false }} />
      {/* The 'camera' screen */}
      <Stack.Screen name="camera" options={{ headerShown: false }} />
      {/* Our new dedicated loader screen */}
      <Stack.Screen name="loader" options={{ headerShown: false }} />
      {/* The 'results' screen */}
      <Stack.Screen name="results" options={{ headerShown: false }} />
    </Stack>
  );
}
