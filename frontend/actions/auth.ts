"use server";

// Auth removed - demo mode only
export const verifyOtp = async (data: {
	email: string;
	otp: string;
	type: string;
}) => {
	return JSON.stringify({ 
		data: null, 
		error: { message: "Auth removed - demo mode only" } 
	});
};
