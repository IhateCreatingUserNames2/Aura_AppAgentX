// aura_voice_client.js

const connectBtn = document.getElementById('connectBtn');
const startRecordBtn = document.getElementById('startRecordBtn');
const stopRecordBtn = document.getElementById('stopRecordBtn');
const disconnectBtn = document.getElementById('disconnectBtn');
const statusTextEl = document.getElementById('statusText');
const chatLogEl = document.getElementById('chat-log');
const remoteAudioEl = document.getElementById('remoteAudio');

// Configuration
const AURA_API_BASE_URL = "http://localhost:8094"; // Ensure this matches your server
const OPENAI_REALTIME_SESSION_ENDPOINT = `${AURA_API_BASE_URL}/openai/rt/session`;
const AURA_CHAT_ENDPOINT = `${AURA_API_BASE_URL}/aura/chat`;
const HEALTH_CHECK_ENDPOINT = `${AURA_API_BASE_URL}/health`; // New for testing

const USER_ID_FOR_AURA = `aura_voice_rt_user_${Math.random().toString(16).slice(2)}`;
let currentAuraSessionId = null;

let pc;
let dc;
let localStream;
let ephemeralKey;

function appendToChatLog(text, type, isError = false) { // Added isError for styling
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', type);
    if (isError) {
        messageDiv.style.color = 'red'; // Simple error styling
    }
    if (type === 'aura-response') {
        const strong = document.createElement('strong');
        strong.textContent = "Aura: ";
        messageDiv.appendChild(strong);
    }
    messageDiv.appendChild(document.createTextNode(text));
    chatLogEl.appendChild(messageDiv);
    chatLogEl.scrollTop = chatLogEl.scrollHeight;
}

function updateStatus(message) {
    console.log("CLIENT STATUS:", message); // Prefixed for clarity
    statusTextEl.textContent = message;
}

function resetControls(isError = false) {
    connectBtn.disabled = isError;
    startRecordBtn.disabled = true;
    stopRecordBtn.disabled = true;
    disconnectBtn.disabled = true; // Should be enabled if connected, disabled otherwise
    if (connectBtn.disabled && !isError) { // If successfully connected
        disconnectBtn.disabled = false;
    }
    if (isError) updateStatus("Error. Ready to reconnect.");
}

async function testServerHealth() {
    console.log("CLIENT: Testing server health at", HEALTH_CHECK_ENDPOINT);
    try {
        const response = await fetch(HEALTH_CHECK_ENDPOINT);
        console.log("CLIENT: Health check response status:", response.status);
        if (response.ok) {
            const data = await response.json();
            console.log("CLIENT: Health check response data:", data);
            updateStatus(`Server health: ${data.status}. Ready to connect.`);
            connectBtn.disabled = false; // Enable connect button if health check passes
            return true;
        } else {
            console.error("CLIENT: Health check failed. Status:", response.status, "Response:", await response.text());
            updateStatus(`Server health check failed: ${response.status}. Check server logs.`);
            appendToChatLog(`Health Check Failed: Status ${response.status}`, "user-transcript", true);
            return false;
        }
    } catch (error) {
        console.error("CLIENT: Error during health check fetch:", error);
        updateStatus("Error connecting to server for health check. Is it running on port 8094?");
        appendToChatLog(`Health Check Network Error: ${error.message}`, "user-transcript", true);
        return false;
    }
}

async function getOpenAIEphemeralKey() {
    updateStatus("Fetching OpenAI session token...");
    console.log("CLIENT: Attempting to fetch ephemeral key from:", OPENAI_REALTIME_SESSION_ENDPOINT);
    try {
        const response = await fetch(OPENAI_REALTIME_SESSION_ENDPOINT, { method: 'POST' });
        console.log("CLIENT: Ephemeral key fetch response status:", response.status);

        if (!response.ok) {
            let errorDetail = `Server error for ephemeral key: ${response.status}`;
            try {
                const errorData = await response.json();
                errorDetail = errorData.detail || errorData.error || JSON.stringify(errorData) || errorDetail;
                console.error("CLIENT: Ephemeral key server error data:", errorData);
            } catch (e) {
                // If response is not JSON, use text
                const errorText = await response.text();
                errorDetail = errorText || errorDetail;
                console.error("CLIENT: Ephemeral key server error text:", errorText);
            }
            throw new Error(errorDetail);
        }

        const data = await response.json();
        console.log("CLIENT: Ephemeral key fetch response data:", data);
        if (data.ephemeral_key) {
            updateStatus("OpenAI session token received.");
            return data.ephemeral_key;
        } else {
            const errorMessage = data.error || "Ephemeral key not in server response.";
            console.error("CLIENT: Error in ephemeral key response structure:", data);
            throw new Error(errorMessage);
        }
    } catch (error) {
        updateStatus(`Error getting token: ${error.message}`);
        appendToChatLog(`Error getting token: ${error.message}`, "user-transcript", true);
        console.error("CLIENT: Full error in getOpenAIEphemeralKey:", error);
        return null;
    }
}

connectBtn.onclick = async () => {
    updateStatus("Connecting...");
    console.log("CLIENT: Connect button clicked.");
    connectBtn.disabled = true;
    // disconnectBtn will be enabled if connection proceeds successfully

    ephemeralKey = await getOpenAIEphemeralKey();
    if (!ephemeralKey) {
        console.error("CLIENT: Failed to get ephemeral key. Aborting connection.");
        resetControls(true);
        return;
    }
    console.log("CLIENT: Ephemeral key obtained:", ephemeralKey.substring(0, 10) + "...");


    pc = new RTCPeerConnection();
    console.log("CLIENT: RTCPeerConnection created.");

    pc.ontrack = (event) => {
        const [stream] = event.streams;
        remoteAudioEl.srcObject = stream;
        updateStatus("Remote audio track received. Aura can speak.");
        console.log("CLIENT: Remote audio stream attached:", stream);
    };

    pc.onicecandidate = (event) => {
        if (event.candidate) {
            console.log("CLIENT: ICE Candidate:", event.candidate);
        } else {
            console.log("CLIENT: All ICE candidates have been gathered.");
        }
    };
    pc.oniceconnectionstatechange = () => {
        updateStatus(`ICE State: ${pc.iceConnectionState}`);
        console.log(`CLIENT: ICE Connection State Change: ${pc.iceConnectionState}`);
        if (pc.iceConnectionState === "connected" || pc.iceConnectionState === "completed") {
            startRecordBtn.disabled = false;
            disconnectBtn.disabled = false; // Enable disconnect now
            updateStatus("Realtime connection established. Ready to speak.");
        } else if (["failed", "disconnected", "closed"].includes(pc.iceConnectionState)) {
            updateStatus(`Connection issue: ${pc.iceConnectionState}.`);
            console.warn(`CLIENT: ICE Connection issue: ${pc.iceConnectionState}.`);
            disconnectCleanup();
        }
    };

    console.log("CLIENT: Creating DataChannel 'oai-events'.");
    dc = pc.createDataChannel("oai-events", { ordered: true });
    dc.onopen = () => { updateStatus("Data channel open."); console.log("CLIENT: Data channel 'oai-events' OPEN.");};
    dc.onmessage = handleOpenAIMessage;
    dc.onclose = () => { updateStatus("Data channel closed."); console.log("CLIENT: Data channel 'oai-events' CLOSED.");};
    dc.onerror = (err) => { updateStatus(`Data channel error: ${err.error ? err.error.message : 'Unknown DC error'}`); console.error("CLIENT: DataChannel Error:", err);};

    try {
        console.log("CLIENT: Requesting microphone access...");
        localStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        console.log("CLIENT: Microphone access granted.");
        localStream.getTracks().forEach(track => pc.addTrack(track, localStream));
        console.log("CLIENT: Local audio track added to PeerConnection.");
    } catch (err) {
        updateStatus("Microphone access error. Please allow microphone.");
        appendToChatLog("Microphone access denied or error.", "user-transcript", true);
        console.error("CLIENT: Microphone access error:", err);
        disconnectCleanup();
        return;
    }

    try {
        console.log("CLIENT: Creating SDP offer...");
        const offer = await pc.createOffer();
        console.log("CLIENT: SDP offer created. Setting local description.");
        await pc.setLocalDescription(offer);
        console.log("CLIENT: Local description set.");

        updateStatus("Sending SDP offer to OpenAI...");
        console.log("CLIENT: Sending SDP offer to OpenAI Realtime API.");
        const sdpResponse = await fetch("https://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview&voice=verse", {
            method: "POST",
            body: offer.sdp,
            headers: { "Authorization": `Bearer ${ephemeralKey}`, "Content-Type": "application/sdp" }
        });
        console.log("CLIENT: SDP offer response status from OpenAI:", sdpResponse.status);

        if (!sdpResponse.ok) {
            const errorText = await sdpResponse.text();
            console.error("CLIENT: OpenAI SDP exchange failed. Status:", sdpResponse.status, "Response:", errorText);
            throw new Error(`OpenAI SDP exchange failed: ${sdpResponse.status} ${errorText}`);
        }
        const answerSdp = await sdpResponse.text();
        console.log("CLIENT: Received SDP answer from OpenAI. Setting remote description.");
        await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });
        updateStatus("SDP exchange complete. Connecting...");
        console.log("CLIENT: Remote description set. WebRTC connection should be establishing.");
        // At this point, oniceconnectionstatechange should eventually hit "connected" or "completed"
    } catch (error) {
        updateStatus(`Realtime setup error: ${error.message}`);
        appendToChatLog(`Realtime setup error: ${error.message}`, "user-transcript", true);
        console.error("CLIENT: Error during WebRTC setup with OpenAI:", error);
        disconnectCleanup();
    }
};

startRecordBtn.onclick = () => {
    if (!dc || dc.readyState !== "open") {
        alert("Data channel not open. Cannot start recording.");
        console.warn("CLIENT: Start record clicked, but DC not open. State:", dc ? dc.readyState : 'null');
        return;
    }
    console.log("CLIENT: Start Record button clicked. Sending 'conversation.item.start'.");
    dc.send(JSON.stringify({ type: "conversation.item.start" }));
    startRecordBtn.disabled = true;
    stopRecordBtn.disabled = false;
    updateStatus("Listening...");
    appendToChatLog("--- Listening... ---", "user-transcript");
};

stopRecordBtn.onclick = () => {
    if (!dc || dc.readyState !== "open") {
        console.warn("CLIENT: Stop record clicked, but DC not open. State:", dc ? dc.readyState : 'null');
        return;
    }
    console.log("CLIENT: Stop Record button clicked. Sending 'conversation.item.stop'.");
    dc.send(JSON.stringify({ type: "conversation.item.stop" }));
    startRecordBtn.disabled = false;
    stopRecordBtn.disabled = true;
    updateStatus("Processing speech...");
};

function disconnectCleanup() {
    console.log("CLIENT: disconnectCleanup called.");
    if (dc) {
        console.log("CLIENT: Closing DataChannel.");
        dc.close();
        dc = null;
    }
    if (pc) {
        console.log("CLIENT: Closing PeerConnection.");
        pc.close();
        pc = null;
    }
    if (localStream) {
        console.log("CLIENT: Stopping local media tracks.");
        localStream.getTracks().forEach(track => track.stop());
        localStream = null;
    }
    ephemeralKey = null;
    resetControls(true);
    updateStatus("Disconnected. Ready to connect.");
    console.log("CLIENT: Disconnect cleanup complete.");
}

disconnectBtn.onclick = disconnectCleanup;

async function handleOpenAIMessage(event) {
    const data = JSON.parse(event.data);
    console.log("CLIENT: OpenAI Event Received:", data);

    if (data.type === "conversation.item.message" && data.item.role === "user" && data.item.is_final) {
        const userTranscript = data.item.content.find(c => c.type === "text")?.text || "[No transcript]";
        appendToChatLog(`You: ${userTranscript}`, "user-transcript");
        updateStatus("Transcript received. Sending to Aura...");
        console.log("CLIENT: Final user transcript:", userTranscript);

        try {
            console.log("CLIENT: Sending transcript to Aura API:", AURA_CHAT_ENDPOINT);
            const auraApiResponse = await fetch(AURA_CHAT_ENDPOINT, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    user_id: USER_ID_FOR_AURA,
                    session_id: currentAuraSessionId,
                    message: userTranscript
                })
            });
            console.log("CLIENT: Aura API response status:", auraApiResponse.status);
            const auraData = await auraApiResponse.json();
            console.log("CLIENT: Aura API response data:", auraData);

            if (auraApiResponse.ok) {
                appendToChatLog(auraData.aura_reply, "aura-response");
                currentAuraSessionId = auraData.session_id;
                updateStatus("Aura replied. Requesting TTS from OpenAI...");
                console.log("CLIENT: Aura reply:", auraData.aura_reply, "New session ID:", currentAuraSessionId);

                if (dc && dc.readyState === "open") {
                    const ttsPayload = {
                        type: "response.create",
                        response: { modalities: ["audio"], text: auraData.aura_reply }
                    };
                    console.log("CLIENT: Sending TTS request to OpenAI via DataChannel:", ttsPayload);
                    dc.send(JSON.stringify(ttsPayload));
                } else {
                    console.warn("CLIENT: DataChannel not open, cannot send TTS request.");
                }
            } else {
                const errorMsg = `Error from Aura API: ${auraData.detail || auraData.error || auraApiResponse.statusText}`;
                appendToChatLog(`Aura Error: ${errorMsg}`, "aura-response", true);
                updateStatus("Error getting response from Aura.");
                console.error("CLIENT: Error from Aura API:", errorMsg, "Full response:", auraData);
            }
        } catch (error) {
            const errorMsg = `Aura API communication failed: ${error.message}`;
            appendToChatLog(`Aura Comm Error: ${errorMsg}`, "aura-response", true);
            updateStatus("Error communicating with Aura.");
            console.error("CLIENT: Aura API communication failed:", error);
        }
    } else if (data.type === "conversation.item.message" && data.item.role === "assistant") {
        if (data.item.is_final) {
             updateStatus("Aura playback complete.");
             console.log("CLIENT: Assistant (Aura) final message/playback complete.");
             startRecordBtn.disabled = false;
             stopRecordBtn.disabled = true;
        } else {
            console.log("CLIENT: Assistant intermediate message:", data.item);
        }
    } else if (data.type === "conversation.item.error") {
        updateStatus(`OpenAI Realtime Error: ${data.error.message}`);
        appendToChatLog(`OpenAI Realtime Error: ${data.error.message}`, "user-transcript", true);
        console.error("CLIENT: OpenAI Event Error:", data.error);
    } else if (data.type === "conversation.item.end") {
        updateStatus("Conversation item ended by OpenAI.");
        console.log("CLIENT: OpenAI 'conversation.item.end' received.");
        startRecordBtn.disabled = false;
        stopRecordBtn.disabled = true;
    }
}

// Initial UI state
console.log("CLIENT: Initializing UI and performing health check.");
connectBtn.disabled = true; // Disable connect until health check
testServerHealth().then(isHealthy => {
    if (isHealthy) {
        resetControls(false); // Enable connect button
        connectBtn.disabled = false;
        statusTextEl.textContent = "Ready to connect.";
    } else {
        resetControls(true); // Keep connect button enabled for retry if health check fails
        statusTextEl.textContent = "Server health check failed. Please check server and refresh.";
    }
});