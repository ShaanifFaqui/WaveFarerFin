import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { FaArrowLeft } from "react-icons/fa";

const defaultPic = "https://cdn-icons-png.flaticon.com/512/149/149071.png";

const UserProfile = () => {
  const navigate = useNavigate();
  const [editMode, setEditMode] = useState(false);
  const [user, setUser] = useState(null);
  const [tempUser, setTempUser] = useState(null);

  useEffect(() => {
    const localUser = JSON.parse(localStorage.getItem("user"));
    const token = localStorage.getItem("token");

    if (localUser?.email && token) {
      fetch(`http://localhost:3000/get-user/${localUser.email}`)
        .then((res) => res.json())
        .then((data) => {
          const profile = {
            name: data.name || "Tourist",
            email: data.email || "tourist@example.com",
            phone: data.phone || "+91 98765 43210",
            location: data.location || "Benaulim, Goa",
            preferredBeaches: data.preferredBeaches || "Palolem Beach, Colva Beach",
            language: data.language || "English",
            profilePic: data.profilePic || defaultPic,
          };
          setUser(profile);
          console.log(profile);
          setTempUser(profile);
        })
        .catch((err) => {
          console.error("Failed to fetch user", err);
        });
    }
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setTempUser((prev) => ({ ...prev, [name]: value }));
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setTempUser((prev) => ({ ...prev, profilePic: reader.result }));
      };
      reader.readAsDataURL(file);
    }
  };

  const handleSave = async () => {
    try {
      const response = await fetch(`http://localhost:3000/update-profile/${tempUser.email}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          phone: tempUser.phone,
          location: tempUser.location,
          preferredBeaches: tempUser.preferredBeaches,
          language: tempUser.language,
          profilePic: tempUser.profilePic,
        }),
      });

      if (!response.ok) throw new Error("Failed to update profile");

      const updatedUser = await response.json();
      setUser(updatedUser);
      setTempUser(updatedUser);
      setEditMode(false);
    } catch (error) {
      console.error("Error updating profile:", error);
    }
  };

  const handleCancel = () => {
    setTempUser({ ...user });
    setEditMode(false);
  };

  if (!user) return <div className="p-6 text-center">Loading profile...</div>;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-100 to-cyan-200 p-6">
      <button
        className="flex items-center gap-2 text-blue-700 font-medium mb-4 hover:underline"
        onClick={() => navigate(-1)}
      >
        <FaArrowLeft /> Back
      </button>

      <div className="max-w-xl mx-auto bg-white rounded-xl shadow-lg p-8">
        <div className="flex flex-col items-center">
          <img
            src={editMode ? tempUser.profilePic : user.profilePic}
            alt="Profile"
            className="w-32 h-32 rounded-full border-4 border-blue-400 object-cover shadow-sm"
          />
          {editMode && (
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="mt-2"
            />
          )}
          <h2 className="text-2xl font-bold mt-4">{user.name}</h2>
          <p className="text-gray-600">{user.email}</p>
        </div>

        <div className="mt-6 space-y-4">
          {[
            ["Phone", "phone"],
            ["Location", "location"],
            ["Preferred Beaches", "preferredBeaches"],
            ["Language", "language"],
          ].map(([label, field]) => (
            <div key={field}>
              <label className="block font-medium">{label}:</label>
              {editMode ? (
                <input
                  name={field}
                  value={tempUser[field]}
                  onChange={handleChange}
                  className="w-full p-2 border rounded"
                />
              ) : (
                <p>{user[field]}</p>
              )}
            </div>
          ))}
        </div>

        <div className="mt-6 text-center">
          {editMode ? (
            <div className="space-x-4">
              <button
                onClick={handleSave}
                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition"
              >
                Save
              </button>
              <button
                onClick={handleCancel}
                className="bg-gray-300 text-black px-4 py-2 rounded hover:bg-gray-400 transition"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={() => setEditMode(true)}
              className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 transition"
            >
              Edit Profile
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default UserProfile;
